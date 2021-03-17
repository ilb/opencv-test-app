#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
#
#
#
#

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys

def help():
	print('python3 rectangle_transform.py [--debug-info] image_name=[name of image]')
	
def do_affine(image,result_name, debug=False):

	img = cv.imread(image)
	rows,cols,ch = img.shape
	if rows >= cols:
		x_max, y_max = cols, rows
	else:
		x_max, y_max = rows, cols
	edges = do_contour(image, debug)
	edges = edges*2
	mx = np.max(edges[0:4,0]) + 1
	my = np.max(edges[0:4,1]) + 1
	new_edges = np.array([[x_max*int(2*e[0]/mx), y_max*int(2*e[1]/my)] for e in edges])

	pts1 = np.float32(edges)
	pts2 = np.float32(new_edges)
	
	mx = np.max(new_edges[0:4,0])
	my = np.max(new_edges[0:4,1])

	M = cv.getPerspectiveTransform(pts1,pts2)
	dst = cv.warpPerspective(img,M,(mx,my))

	if debug:
		plt.subplot(121),plt.imshow(img),plt.title('Input')
		plt.subplot(122),plt.imshow(dst),plt.title('Output')
		plt.show()	

	cv.imwrite(result_name,dst)
	
def do_contour(image, debug=False):

	im = cv.imread(image)
	rows,cols,ch = im.shape
	im_resize = cv.resize(im,(cols//2,rows//2))
	img = cv.cvtColor(im_resize, cv.COLOR_BGR2GRAY)
	canny = do_canny(img, debug)
	kernel = np.ones((5, 5))
	canny_dil = cv.dilate(canny, kernel, iterations=2)
	canny_result = cv.erode(canny_dil, kernel, iterations=1)
	ret, thresh = cv.threshold(canny_result, 127, 255, 0)
	contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	cnts = sorted([[cnt, len(cnt)] for cnt in contours], key = lambda c: c[1])
	cnt = cv.approxPolyDP(cnts[-1][0], 0.02 * cv.arcLength(cnts[-1][0], True), True)
	
	if debug:
		cv.imshow('Canny Edges After Contouring', canny_result)
		cv.waitKey(0) 
		cv.drawContours(img, cnt, -1, (0, 255, 0), 3) 
  
		cv.imshow('Contours', img)
		cv.waitKey(0) 
		cv.destroyAllWindows()
	
	return cnt.reshape((4,2))

def do_canny(img, debug=False):
	
	#blur = cv.GaussianBlur(img, (5,5), 2)
	edges = cv.Canny(img,50,150)
	
	if debug:
		plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])	
		plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	
		plt.show()
	
	return edges
	
if __name__ == '__main__':

	try:
		if sys.argv[-1][0:10] == 'image_name':
			exec(sys.argv[-1].replace('=','="') + '"')
		debug = sys.argv[1] == '--debug-info'
	except Exception:
		help()
		exit(1)
		
	result_name = 'new_' + image_name.split('.')[0] + '.png'
	do_affine(image_name, result_name, debug)

