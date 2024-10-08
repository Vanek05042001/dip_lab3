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

clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
rgb_result_image = np.empty(np.shape(image1), np.uint8)
rgb_result_image[:, :, 0] = clahe.apply(image1[:, :, 0])
rgb_result_image[:, :, 1] = clahe.apply(image1[:, :, 1])
rgb_result_image[:, :, 2] = clahe.apply(image1[:, :, 2])

#result_image = cv.filter2D(cv.medianBlur(cv.filter2D(image1, -1, kernel3), 5), -1, kernel77)
#result_image = cv.filter2D(cv.medianBlur(cv.filter2D(gray_image1, -1, kernel3), 5), -1, kernel77)
#result_image = cv.filter2D(cv.medianBlur(image1, 3), -1, kernel3)
#result_image = cv.filter2D(cv.medianBlur(image1, 3), -1, kernel2)

result_image = cv.filter2D(rgb_result_image, -1, kernel2)

plt.subplot(332)
plt.xticks([]), plt.yticks([])
plt.title('Фильтрация')
plt.imshow(result_image, 'gray')

plt.show()


# =============================================================================
# median_image1 = cv.medianBlur(image1, 3)
# median_image2 = cv.medianBlur(image1, 5)
# 
# 
# plt.subplot(gs[1])
# plt.imshow(median_image1, cmap='gray')
# psnr = util.getPSNR(image1, median_image1)
# ssim = util.getSSIM(image1, median_image1)
# plt.title(f'Восстановленное изображение \n '
#           f'медианным фильтром 3х3 \n'
#           f'PSNR = {psnr:.3f} \n SSIM = {ssim:.3f}')
# 
# plt.subplot(gs[2])
# plt.imshow(median_image2, cmap='gray')
# psnr = util.getPSNR(image1, median_image2)
# ssim = util.getSSIM(image1, median_image2)
# plt.title(f'Восстановленное изображение \n '
#           f'медианным фильтром 5х5 \n'
#           f'PSNR = {psnr:.3f} \n SSIM = {ssim:.3f}')
# 
# plt.show()
# =============================================================================


###################################
####################################
####################################
####################################
####################################


# =============================================================================
# kernel55 = np.ones((5, 5), np.float32) / 25
# kernel77 = np.ones((7, 7), np.float32) / 49
# 
# filtered_image1 = cv.filter2D(image1, -1, kernel55)
# filtered_image2 = cv.filter2D(image1, -1, kernel77)
# gaussian_image1 = cv.GaussianBlur(image1, (7, 7), 0)
# gaussian_image2 = cv.GaussianBlur(image1, (15, 15), 0)
# 
# gs = plt.GridSpec(2, 3)
# plt.figure(figsize=(15, 12))
# 
# plt.subplot(gs[0, 0])
# plt.xticks([]), plt.yticks([])
# plt.title('Исходное изображение')
# plt.imshow(image1, cmap='gray')
# 
# plt.subplot(gs[0, 1])
# plt.xticks([]), plt.yticks([])
# plt.title(f'Результат средней линейной \n фильрации с ядром 5х5 \n '
#           f'PSNR = {util.getPSNR(image1, filtered_image1):.3f} \n '
#           f'SSIM = {util.getSSIM(image1, filtered_image1):.3f}')
# plt.imshow(filtered_image1, 'gray')
# 
# plt.subplot(gs[0, 2])
# plt.xticks([]), plt.yticks([])
# plt.title(f'Результат средней линейной \n фильрации с ядром 7х7 \n '
#           f'PSNR = {util.getPSNR(image1, filtered_image2):.3f} \n '
#           f'SSIM = {util.getSSIM(image1, filtered_image2):.3f}')
# plt.imshow(filtered_image2, 'gray')
# 
# plt.subplot(gs[1, 1])
# plt.xticks([]), plt.yticks([])
# plt.title(f'Результат гауссовской  \n фильрации с ядром 7х7 \n '
#           f'PSNR = {util.getPSNR(image1, gaussian_image1):.3f} \n '
#           f'SSIM = {util.getSSIM(image1, gaussian_image1):.3f}')
# plt.imshow(gaussian_image1, 'gray')
# 
# plt.subplot(gs[1, 2])
# plt.xticks([]), plt.yticks([])
# plt.title(f'Результат гауссовской  \n фильрации с ядром 15х15 \n '
#           f'PSNR = {util.getPSNR(image1, gaussian_image2):.3f} \n '
#           f'SSIM = {util.getSSIM(image1, gaussian_image2):.3f}')
# plt.imshow(gaussian_image2, 'gray')
# 
# plt.show()
# 
# 
# ####################################
# ####################################
# ####################################
# ####################################
# ####################################
# 
# kernel1 = np.asarray([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# kernel2 = np.asarray([[-0.25, -0.25, -0.25], [-0.25, 3, -0.25], [-0.25, -0.25, -0.25]])
# kernel3 = np.asarray([[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]])
# 
# image1_median = cv.medianBlur(image1, 3)
# 
# filtered_image1 = cv.filter2D(image1, -1, kernel1)
# filtered_image1_median = cv.filter2D(image1_median, -1, kernel1)
# filtered_image2 = cv.filter2D(image1, -1, kernel2)
# filtered_image2_median = cv.filter2D(image1_median, -1, kernel2)
# filtered_image3 = cv.filter2D(image1, -1, kernel3)
# filtered_image3_median = cv.filter2D(image1_median, -1, kernel3)
# 
# # вывод
# plt.figure(figsize=(15, 17))
# 
# plt.subplot(334)
# plt.xticks([]), plt.yticks([])
# plt.title(f'Исходное изображение отфильтрованное \n медианным фильтром \n'
#           f'PSNR = {util.getPSNR(image1, image1_median):.3f} \n '
#           f'SSIM = {util.getSSIM(image1, image1_median):.3f}')
# plt.imshow(image1_median, cmap='gray')
# 
# plt.subplot(332)
# plt.xticks([]), plt.yticks([])
# plt.title(f'Свертка изображения с ядром 1 \n'
#           f'PSNR = {util.getPSNR(image1, filtered_image1):.3f} \n '
#           f'SSIM = {util.getSSIM(image1, filtered_image1):.3f}')
# plt.imshow(filtered_image1, 'gray')
# 
# plt.subplot(333)
# plt.xticks([]), plt.yticks([])
# plt.title(f'Свертка отфильтрованного изображения с ядром 1 \n'
#           f'PSNR = {util.getPSNR(image1, filtered_image1_median):.3f} \n '
#           f'SSIM = {util.getSSIM(image1, filtered_image1_median):.3f}')
# plt.imshow(filtered_image1_median, 'gray')
# 
# plt.subplot(335)
# plt.xticks([]), plt.yticks([])
# plt.title(f'Свертка изображения с ядром 2 \n'
#           f'PSNR = {util.getPSNR(image1, filtered_image2):.3f} \n '
#           f'SSIM = {util.getSSIM(image1, filtered_image2):.3f}')
# plt.imshow(filtered_image2, 'gray')
# 
# # =============================================================================
# # plt.subplot(336)
# # plt.xticks([]), plt.yticks([])
# # plt.title(f'Свертка изображения с ядром 3 \n'
# #           f'PSNR = {util.getPSNR(image1, filtered_image3):.3f} \n '
# #           f'SSIM = {util.getSSIM(image1, filtered_image3):.3f}')
# # plt.imshow(filtered_image3, 'gray')
# # =============================================================================
# 
# plt.show()
# 
# 
# gamma = 1.05
# lut = lambda i: i ** gamma
# image2 = lut(gray_image1)
# 
# lut = lambda i: 255 * ((i - np.min(i)) / (np.max(i) - np.min(i)))
# image3 = lut(gray_image1)
# =============================================================================
