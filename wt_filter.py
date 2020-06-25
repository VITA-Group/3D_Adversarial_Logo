#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:03:39 2019

@author: zhouge
"""

import cv2

from pywt import dwt2, idwt2

img = cv2.imread('/home/zhouge/Pictures/boat.png',0)

m,n=img.shape
img = img / 255

p,(q,r,s)=dwt2(img,'db2')
def wt_filter(img, dtype = 'rgb'):
    img = img / 255
    import numpy as np
    def thres(img):
        x, y = img.shape
        for i in range(x):
            for j in range(y):
                if img[i][j] >0.11:
                    img[i][j] = 0.
                else:
                    img[i][j] = 1.
                

    thres(p)
    thres(q)
    thres(r)
    thres(s)
    return p*255, q*255, r*255, s*255

with open('/home/zhouge/Documents/cifar/data/pic_1000_origin(1).pkl','rb') as file:
    data = pickle.load(file)
    
    for i in range(1000):
        
        img = data[i][0]
        label = data[i][1]
        freq_l, freq_h = image_filter(radius, img)
        low_freq[i] = [freq_l,label]
#        low_freq.update(l_freq)
        high_freq[i] = [freq_h,label]
#        original[i] = [img, ]
#        high_freq.update(h_freq)
#        plt.subplot(131),plt.imshow(img),plt.title('origial')
#        plt.subplot(132),plt.imshow(freq_l),plt.title('lowfreq')
#        plt.subplot(133),plt.imshow(freq_h),plt.title('highfreq')
        print(i)

#cv2.imwrite('/home/zhouge/Pictures/wt_r.png',np.uint8(r*255))
#cv2.imwrite('/home/zhouge/Pictures/wt_s.png',np.uint8(s*255))