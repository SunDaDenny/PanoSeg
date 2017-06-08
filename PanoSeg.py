import os
import cv2
import sys
import numpy as np

input_file = sys.argv[1]
input_img = cv2.imread(input_file,1)
height, width = input_img.shape[:2]
face_size = width/4

output_dir = sys.argv[2]
cube_path = output_dir + '/cube/'

yolo_path = 'darknet/'
yolo_result_path = output_dir + '/detection'

mask_result_path = output_dir + '/mask/'


#Panorama to cube
print('Start panorama to cube')

cmd_pano_cube = 'sphere2cube ' + input_file + ' -r ' + str(face_size) + ' -f PNG -o ' + cube_path + '_;'
cmd_pano_cube_45 = 'sphere2cube ' + input_file + ' -r ' + str(face_size) + ' -f PNG -o ' + cube_path + '45_' + ' -R 0 0 -45'
os.system(cmd_pano_cube + cmd_pano_cube_45)


#Rename all face image and rotate top/down face
print('Rename all face image and rotate top/down face')

cube_list = open( cube_path + 'list.txt','w') 

for i in range(1,7):
	cmd_rename = 'mv ' + cube_path + '_000' + str(i) + '.png ' + cube_path + str(2*i-1) + ';'
	cmd_rename_45 = 'mv ' + cube_path + '45_000' + str(i) + '.png ' + cube_path + str(2*i)
	os.system(cmd_rename + cmd_rename_45)
	cube_list.write('../' + cube_path + str(2*i-1) + '\n' + '../' + cube_path + str(2*i) + '\n') 
cube_list.close()

for i in range(9,13):
	face_img = cv2.imread(cube_path + str(i),1)
	(h, w) = face_img.shape[:2]
	M = cv2.getRotationMatrix2D((w / 2, h / 2), 180, 1.0)
	rotated = cv2.warpAffine(face_img, M, (w, h))
	cv2.imwrite( cube_path + str(i) + '.png', rotated )
	os.system('mv ' + cube_path + str(i) + '.png ' + cube_path + str(i))


#YOLO detection
print('Start YOLO detection')

if not os.path.exists(yolo_result_path):
    os.makedirs(yolo_result_path)

cmd_cd_in = 'cd ' + yolo_path + ';'
cmd_detector = './darknet detect cfg/yolo.cfg pretrain/yolo.weights ' + '../' + cube_path + 'list.txt' + ' -out ' + '../' + yolo_result_path + ';' 
cmd_cd_out = 'cd ..'
os.system(cmd_cd_in + cmd_detector + cmd_cd_out)


#Crop and resize bbox image
print('Start crop and resize bbox image')

mask_bbox_scale = 1.25

for i in range(1,13):
	if not os.path.exists(yolo_result_path + '/' + str(i)) :
		os.makedirs(yolo_result_path + '/' + str(i))
	face_img = cv2.imread(cube_path + str(i),1)
	face_h, face_w = face_img.shape[:2]

	with open(yolo_result_path + '/' + str(i) + '_bbox.txt') as fp:
		bboxs = fp.read().splitlines()

	for j, val in enumerate(bboxs):
		bbox = val.split()
		for k in range(2,6):
			bbox[k] = float(bbox[k])

		center = [ (bbox[2]+bbox[3])/2 , (bbox[4]+bbox[5])/2 ]
		bbox_size = [ (bbox[3]-bbox[2]) * mask_bbox_scale , (bbox[5]-bbox[4]) * mask_bbox_scale ]
		bbox[2] = int(center[0] - bbox_size[0]/2)
		bbox[3] = int(center[0] + bbox_size[0]/2)
		bbox[4] = int(center[1] - bbox_size[1]/2)
		bbox[5] = int(center[1] + bbox_size[1]/2)

		if bbox[2] < 0 : bbox[2] = 0
		if bbox[3] > face_w : bbox[3] = face_w
		if bbox[4] < 0 : bbox[4] = 0
		if bbox[5] > face_h : bbox[5] = face_h
		obj_img = face_img[ bbox[4]:bbox[5], bbox[2]:bbox[3]]
		#cv2.imshow('test' + str(i) + '_' + str(j) , obj_img)
		cv2.imwrite( yolo_result_path + '/' + str(i) + '/' + str(j) + '.png', obj_img )

		bbox_scale_list = open( yolo_result_path + '/' + str(i) + '/bbox_scale.txt','a') 
		bbox_scale_list.write( bbox[0] + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' ' + str(bbox[4]) + ' ' + str(bbox[5]) +'\n')
		bbox_scale_list.close()

#FastMask
print('Start FastMask detection')

for i in range(1,13):
	j = 0
	while(1):
		roi_img_path = yolo_result_path + '/' + str(i) + '/' + str(j) + '.png'
		if(os.path.exists(roi_img_path)):
			#print(roi_img_path)
			cmd_cd_in = 'cd FastMask;'
			cmd_mask = 'python2 mask_detect.py 0 fm-res39 ../' + roi_img_path + ' --init_weights=fm-res39_final.caffemodel --threshold=0.9 --out=../' + yolo_result_path + '/' + str(i) + '/' + str(j) + '_mask.png;'
			cmd_cd_out = 'cd ..'
			os.system(cmd_cd_in + cmd_mask + cmd_cd_out)
			j+=1
		else:
			break

#Paste mask back to face image and merge the result to panorama format 
print('Start paste mask back to face image\nand merge the result to panorama format')

label_list = ['chair','sofa','bed','diningtable','tvmonitor','refrigerator','pottedplant']
#			'person','bottle','laptop','book','clock','bowl']

if not os.path.exists(mask_result_path):
    os.makedirs(mask_result_path)

for i, label in enumerate(label_list):

	mask_label_path = mask_result_path + '/' + label
	if not os.path.exists(mask_label_path):
   		os.makedirs(mask_label_path)

   	mask_img = []
   	for j in range(0,12):
   		mask_img.append(np.zeros([ face_size, face_size, 1], dtype=np.uint8))

   	for j in range(1,13):
   		bbox_scale_file = yolo_result_path + '/' + str(j) + '/bbox_scale.txt'
   		if(os.path.exists(bbox_scale_file)):
	   	
	   		with open(bbox_scale_file) as fp:
	   			bboxs = fp.read().splitlines()

	   		for k, val in enumerate(bboxs):
				bbox = val.split()
				for l in range(2,6):
					bbox[l] = int(bbox[l])
				if bbox[0] == label:
					bbox_mask_path = yolo_result_path + '/' + str(j) + '/' + str(k) + '_mask.png'
					if(os.path.exists(bbox_mask_path)) :
						bbox_mask = cv2.imread(bbox_mask_path ,0)
						mask_img[j-1][bbox[4] : bbox[5] , bbox[2] : bbox[3] , 0] = bbox_mask + mask_img[j-1][bbox[4] : bbox[5] , bbox[2] : bbox[3] , 0]
		
		cv2.imwrite( mask_label_path + '/' + str(j) + '.png', mask_img[j-1] )

	cmd_cd_in = 'cd ' + mask_label_path + ';'
	cmd_mask_cube_pano = 'cube2sphere 1.png 5.png 3.png 7.png 11.png 9.png -r  ' + str(width) + ' ' + str(height) + ' -fPNG -omask_;'
	cmd_mask_cube_pano_45 = 'cube2sphere 2.png 6.png 4.png 8.png 12.png 10.png -r  ' + str(width) + ' ' + str(height) + ' -fPNG -omask45_ -R 0 0 45;'
	cmd_cd_out = 'cd ../../..;'
	os.system(cmd_cd_in + cmd_mask_cube_pano + cmd_mask_cube_pano_45 + cmd_cd_out)

	pano_mask_img = cv2.imread(mask_label_path + '/mask_0001.png' ,0)
	pano_mask_45_img = cv2.imread(mask_label_path + '/mask45_0001.png' ,0)
	merge = pano_mask_img + pano_mask_45_img
	merge[merge>0] = 255
	cv2.imwrite( mask_label_path + '.png', merge )

	print('class ' + str(i) + ' ' + label + ' finish')


#Map all result mask to original image
print('Start map all result mask to original image')

def im2double(im):
    info = np.iinfo(im.dtype)
    return im.astype(np.float) / info.max

def im2int(im):
	im = im * 255
	return im.astype(np.int)

COLORS = [0xAEEE00, 0xAEEE00, 0xF977D2, 0xEFC94C,
		0x468966, 0xA7A37E, 0x00A545, 0x046380, 
		0xE6E2AF, 0xB64926, 0x8E2800, 0xFFE11A,
		0xFF6138, 0x193441, 0xFF9800, 0x7D9100]

image =  cv2.imread(input_file ,1)
image = im2double(image)

for i, label in enumerate(label_list):

	mask_label_path = mask_result_path + label
	if(os.path.exists(mask_label_path + '.png')):
		mask = cv2.imread( mask_label_path + '.png',0)
		mask = im2double(mask)

		mask[mask > 0] = 0.5
		mask[mask == 0] = 1
		color = COLORS[i % len(COLORS)]

		#print(np.count_nonzero(mask))
		for k in range(3):
			image[:,:,k] = image[:,:,k] * mask

		mask[mask == 1] = 0
		mask[mask > 0] = 0.5
		
		for k in range(3):
			image[:,:,k] += mask * (color & 0xff)/255
			color >>= 8;

cv2.imwrite(output_dir + '/final.png' , im2int(image) )
cv2.imshow('img' , image)
cv2.waitKey(1000000)
