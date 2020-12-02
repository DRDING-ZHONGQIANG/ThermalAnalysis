# Author: Dr. DING Zhongqiang
#
# By downloading, copying, installing or using the software you agree to this license. 
# If you do not agree to this license, do not download, install, copy or use the software.
#
# Copyright (C) 2000-2020, Intel Corporation, all rights reserved.
# Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
# Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
# Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
# Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
# Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
# Copyright (C) 2019-2020, Xperience AI, all rights reserved.
# Third party copyrights are property of their respective owners.
#
# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall copyright holders or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.
#
#!/usr/bin/env python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import codecs
import math

def gaussian_blur(img, kernel_size, average, sigma):
    
    if sigma == -1:
        sd = math.sqrt(kernel_size)
    else:
        sd = sigma    
    kernel_1D = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    for i in range(kernel_size):
        kernel_1D[i] = 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((kernel_1D[i] - 0) / sd, 2) / 2)      
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
    
    if len(img.shape) == 3:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
    image_row, image_col = img.shape
    kernel_row, kernel_col = kernel_2D.shape
    output = np.zeros(img.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = img
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel_2D * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel_2D.shape[0] * kernel_2D.shape[1]    
    return output

def produce_img(path, filename, filename_suffix, scalar) :   
   
    infile = os.path.join(path, filename + "." + filename_suffix)      
    with codecs.open(infile, encoding='utf-8-sig') as f:
        data = np.loadtxt(f)    
        
    filename_suffix = 'jpg'    
    outfile = os.path.join(path, filename + "." + filename_suffix) 
    data = data * scalar
    cv2.imwrite(outfile, data)
    return

def produce_edge(path, filename, filename_suffix, kernel_size, l_threshold, h_threshold, sigma):

    infile = os.path.join(path, filename + "." + filename_suffix)      
    img    = cv2.imread(infile,0)     
    blur  = gaussian_blur(img, kernel_size, True, sigma)  
    outfile  = os.path.join(path, filename + "_blur." + filename_suffix)  
    cv2.imwrite(outfile, blur)    
    canny = cv2.Canny(np.uint8(blur), l_threshold, h_threshold) 
    outfile  = os.path.join(path, filename + "_canny." + filename_suffix)  
    cv2.imwrite(outfile, canny)   
    return

def main():
 
    path = 'C:\\projects\\IR_data' 
    
    filename = 'unit4_0_028V_15mA_100s_Ampl';  filename_suffix = 'asc' 
    produce_img(path, filename, filename_suffix, 256)
    
    filename_suffix = 'jpg' ;  l_threshold = 5;  h_threshold = 20; kernel_size = 15
    produce_edge(path, filename,filename_suffix, kernel_size, l_threshold, h_threshold,-1)    
    
    filename = 'unit4_0_028V_15mA_100s_phase';   filename_suffix = 'asc' 
    produce_img(path, filename, filename_suffix, 1)  
    
    filename_suffix = 'jpg' ;  l_threshold = 5 ; h_threshold = 15; kernel_size = 21
    produce_edge(path, filename,filename_suffix, kernel_size, l_threshold, h_threshold,-1)        
    
    return
   
if __name__ == '__main__':
    main()