# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 23:04:16 2024

@author: Vanya
"""

import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util

image1 = cv.imread('3l.jpg')
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
rgb_image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)

gs = plt.GridSpec(2, 2)
plt.figure(figsize=(10, 4))
plt.subplot(gs[0])
plt.imshow(rgb_image1)
plt.title('3l.jpg')
plt.xticks([]), plt.yticks([])

kernel55 = np.ones((5, 5), np.float32) / 25
kernel77 = np.ones((7, 7), np.float32) / 49

kernel1 = np.asarray([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
kernel2 = np.asarray([[-0.25, -0.25, -0.25], [-0.25, 3, -0.25], [-0.25, -0.25, -0.25]])
kernel3 = np.asarray([[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]])


# Эквализация
clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
result_image = np.empty(np.shape(image1), np.uint8)
result_image[:, :, 0] = clahe.apply(image1[:, :, 0])
result_image[:, :, 1] = clahe.apply(image1[:, :, 1])
result_image[:, :, 2] = clahe.apply(image1[:, :, 2])

result_image = cv.filter2D(result_image, -1, kernel3)

plt.subplot(332)
plt.xticks([]), plt.yticks([])
plt.title('Фильтрация')
plt.imshow(result_image, 'gray')

plt.show()