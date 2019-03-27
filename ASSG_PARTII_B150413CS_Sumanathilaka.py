# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:21:13 2019

@author: desha-
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    
    #Reading the grayscale image
    Img = cv2.imread('C:/Users/desha/Desktop/cameraman.tif', cv2.IMREAD_GRAYSCALE) 
   
    #size of the image in pixels
    img_height, img_width = np.shape(Img) 
    print("Image Size =", img_height, " X ", img_width)
    
    #plot original image
    plt.imshow(Img, cmap=plt.get_cmap('gray')) 
    plt.title("Original Image ")
    plt.show()
    array_L=[0]*256
    file = open("img.txt", "w")

    for px in range(0, img_height):
        for py in range(0, img_width):
            array_val=array_L[Img[px][py]]+1
            array_L[Img[px][py]]=array_val
            file.write (str(Img[px][py]) + " ")
        file.write("\n")

##Normalization##        
no_of_pixels=img_height*img_width
for x in range (0, 256):
    array_L[x]= array_L[x]/no_of_pixels

##cumulative Sum##
array_cumulative_sum=[0]*256
sum=array_L[0]
for x in range (0, 255):
    array_cumulative_sum[x]=sum
    sum=sum+array_L[x+1]
array_cumulative_sum[255]=sum


##cumulative Mean## 
cumulative_means=[0]*256
sum=0
for x in range (0, 255):
    cumulative_means[x]=sum
    sum=sum+((x+1)*array_L[x+1])
cumulative_means[255]=sum 
 
##global Intensioty Mean##                
global_intensity_mean=0
for x in range (0, 256):
    global_intensity_mean=global_intensity_mean+(x*array_L[x])


##class variance##
threshold=0
k2=0
class_var_array= [None]*256
for x in range(0,255):
   class_variance=global_intensity_mean*array_cumulative_sum[x]
   class_variance_1=class_variance - cumulative_means[x]
   class_variance1=class_variance_1*class_variance_1
   class_variance2=1- array_cumulative_sum[x]
   class_variance3= array_cumulative_sum[x] * class_variance2
   class_variance4=class_variance1/class_variance3
   class_var_array[x]=class_variance4
   if threshold<class_variance4:
      threshold=class_variance4
      
sum_index=0
count_index=0

for x in range(0,255):
    if threshold==class_var_array[x]:
        sum_index=sum_index+x
        count_index=count_index+1
        
final_index=sum_index/count_index
final_index=round(final_index)


##obtaining Seperability measure
SigmaG =0

for x in range (0,255):
      SigmaG1 = (x-global_intensity_mean)**2
      SigmaG=SigmaG + (SigmaG1*array_cumulative_sum[x])


separability_measure=class_var_array[final_index] / SigmaG


print("Seperability Measure :",separability_measure )  

for i in np.arange(img_height):
          for j in np.arange(img_width):
            a = Img.item(i,j)
            if a > final_index:           #Final Threshold
               b=255                    #initializing value G
               Img.itemset((i,j) ,b)      # G(i,j) = 255 , if Img(i,j) > T
            else:
               b=0                     #  G(i,j)= 0  , if Img(i,j)  <=T
               Img.itemset((i,j) ,b)
     # final image plot           
plt.imshow(Img, cmap=plt.get_cmap('gray'))    
plt.title("Resulting Image 1")
plt.show()


################################################part2#########################
   

##Seperability measure#
Img = cv2.imread('C:/Users/desha/Desktop/cameraman.tif', cv2.IMREAD_GRAYSCALE) 

##adding Guassian noise

mean = 0
var = 0.1
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, (229, 229)) #  np.zeros((224, 224), np.float32)

noisy_image = np.zeros(Img.shape, np.float32)

if len(Img.shape) == 2:
    noisy_image = Img + gaussian
else:
    noisy_image[:, :, 0] = Img[:, :, 0] + gaussian
    noisy_image[:, :, 1] = Img[:, :, 1] + gaussian
    noisy_image[:, :, 2] = Img[:, :, 2] + gaussian

cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
Img = noisy_image.astype(np.uint8)

plt.imshow(Img, cmap=plt.get_cmap('gray')) 
plt.title("Noisy Image ")
plt.show()

img_height, img_width = np.shape(Img) 
array_L=[0]*256
file = open("img1.txt", "w")

for px in range(0, img_height):
      for py in range(0, img_width):
            array_val=array_L[Img[px][py]]+1
            array_L[Img[px][py]]=array_val
            file.write (str(Img[px][py]) + " ")
      file.write("\n")


##Normalization##        
no_of_pixels=img_height*img_width
for x in range (0, 256):
    array_L[x]= array_L[x]/no_of_pixels

##cumulative Sum##
array_cumulative_sum=[0]*256
sum=array_L[0]
for x in range (0, 255):
    array_cumulative_sum[x]=sum
    sum=sum+array_L[x+1]
array_cumulative_sum[255]=sum


##cumulative Mean## 
cumulative_means=[0]*256
sum=0
for x in range (0, 255):
    cumulative_means[x]=sum
    sum=sum+((x+1)*array_L[x+1])
cumulative_means[255]=sum 
 
##global Intensioty Mean##                
global_intensity_mean=0
for x in range (0, 256):
    global_intensity_mean=global_intensity_mean+(x*array_L[x])


##class variance##
threshold=0
k2=0
class_var_array= [None]*256
for x in range(0,255):
   class_variance=global_intensity_mean*array_cumulative_sum[x]
   class_variance_1=class_variance - cumulative_means[x]
   class_variance1=class_variance_1*class_variance_1
   class_variance2=1- array_cumulative_sum[x]
   class_variance3= array_cumulative_sum[x] * class_variance2
   class_variance4=class_variance1/class_variance3
   class_var_array[x]=class_variance4
   if threshold<class_variance4:
      threshold=class_variance4
      
sum_index=0
count_index=0

for x in range(0,255):
    if threshold==class_var_array[x]:
        sum_index=sum_index+x
        count_index=count_index+1
        
final_index=sum_index/count_index
final_index=round(final_index)


##obtaining Seperability measure
SigmaG =0

for x in range (0,255):
      SigmaG1 = (x-global_intensity_mean)**2
      SigmaG=SigmaG + (SigmaG1*array_cumulative_sum[x])


separability_measure=class_var_array[final_index] / SigmaG


print("Seperability Measure :",separability_measure )     




for i in np.arange(img_height):
          for j in np.arange(img_width):
            a = Img.item(i,j)
            if a > final_index:           #Final Threshold
               b=255                    #initializing value G
               Img.itemset((i,j) ,b)      # G(i,j) = 255 , if Img(i,j) > T
            else:
               b=0                     #  G(i,j)= 0  , if Img(i,j)  <=T
               Img.itemset((i,j) ,b)
     # final image plot           
plt.imshow(Img, cmap=plt.get_cmap('gray'))    
plt.title("After adding Guassian Image -  Result")
plt.show()
