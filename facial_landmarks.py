# -*- coding: utf-8 -*-

from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

## 获取人脸训练模型
# 获取模型路径
shape_path = 'shape_predictor_68_face_landmarks.dat'
# dlib.get_frontal_face_detector() 返回 fhog_object_detector 类，需要输入numpy ndarray类型8位灰色图像或者彩色图像
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_path)

for  p_num in range(29):
	# 获取图片路径
	img_path = 'picture/' + str(p_num+1) +'.jpg'
	print('第{}张图片'.format(p_num+1))
	# 读取图片
	image = cv2.imread(img_path)
	# 修改图片width为 500，height按比例缩放
	#image = imutils.resize(image,width=500)
	# 将灰度图像转为RGB图像
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	# 获取人脸68个地标
	rects = detector(gray,1)

	for (i,rect) in enumerate(rects):
		shape = predictor(gray,rect)
		shape = face_utils.shape_to_np(shape)

		(x,y,w,h) = face_utils.rect_to_bb(rect)
		# 框出人脸
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
		# 标出人脸数目
		cv2.putText(image,'Face #{}'.format(i+1),(x-10,y-10),
			cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
		# 标出人脸地标
		for (x,y) in shape:
			cv2.circle(image,(x,y),1,(0,0,255),-1)

	cv2.imshow('output',image)
	cv2.waitKey(0)