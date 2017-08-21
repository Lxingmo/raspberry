# -*- coding: utf-8 -*-

from imutils.face_utils import FaceAligner
from imutils.face_utils import  rect_to_bb
import imutils
import dlib
import cv2


## 获取人脸训练模型,初始化了基于HOG的检测器
# 获取模型路径
shape_path = '/home/xingmo/code/code/dlib/shape_predictor_68_face_landmarks.dat'
# dlib.get_frontal_face_detector() 返回 fhog_object_detector 类，需要输入numpy ndarray类型8位灰色图像或者彩色图像
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_path)
fa = FaceAligner(predictor,desiredFaceWidth=256)

for p_num in range(29):
	img_path = '/home/xingmo/code/code/dlib/picture/' + str(p_num+1) + '.jpg'
	print('第{}张图片'.format(p_num+1))
	# 读取图片
	image = cv2.imread(img_path)
	# 修改图片width为 500，height按比例缩放
	#image = imutils.resize(image,width=800)
	# 将灰度图像转为RGB图像
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	cv2.imshow('input',image)

	# 获取人脸68个地标
	rects = detector(gray,1)

	for (i,rect) in enumerate(rects):
		(x,y,w,h) = rect_to_bb(rect)
		faceOrig = imutils.resize(image[y:y+h,x:x+w],width=256)
		faceAligner = fa.align(image,gray,rect)

	cv2.imshow('Original',faceOrig)
	cv2.imshow('Aligned',faceAligner)
	cv2.waitKey(0)