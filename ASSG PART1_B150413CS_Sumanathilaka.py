# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:56:29 2019

@author: deshan
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    ###### Step 1 ######
    
    I = cv2.imread('C:/Users/desha/Desktop/1.jpg', cv2.IMREAD_GRAYSCALE)  # Read a grayscale image
   
    height, width = np.shape(I) #size of the image (no of pixels)
    print("Size of Image: ", height, " X ", width)
    plt.imshow(I, cmap=plt.get_cmap('gray')) #plot original image
    plt.title("Original Image (Step 1)")
    plt.show()
    f = open("img.txt", "w")
    sum = 0
    for px in range(0, height):
        for py in range(0, width):
            sum = sum + I[px][py]     #sum of all pixels
            f.write (str(I[px][py]) + " ")
        f.write("\n")
   
    no_of_pixels=height*width  #no of pixels
    Threshold =sum/no_of_pixels # initial Threshold
    
    print("Threshold : ", Threshold) 
    
    ##### Step2 #####
    sum1=0
    sum2=0
    count1=0
    count2=0
 
    for i in np.arange(height):
        for j in np.arange(width):
            a = I.item(i,j)
   #Dividing the pixels into two groups based on pixel values 
            if a > Threshold:   
                sum1 =sum1 + a
                count1 =count1 + 1
               
            else:
                sum2 =sum2 + a
                count2 =count2 + 1
              
               
    

    A1 =sum1/count1
    A2 =sum2/count2
    print("Average G1 :",A1 )
    print("Average G2 :",A2 )
     
    # calculating New threshold
    Threshold =(1/2)*(A1+A2)
    c= 2
    
    #### STEP3-6 #####
    
    while c > 1:  #checking for T0 Condition
        sum1=0
        sum2=0
        count1=0
        count2=0
        Threshold2=Threshold
     
        for i in np.arange(height):
          for j in np.arange(width):
            a = I.item(i,j)
    
            if a > Threshold:
                sum1 =sum1 + a
                count1 =count1 + 1
                
            else:
                sum2 =sum2 + a
                count2 =count2 + 1
               
              
        
        A1 =sum1/count1
        A2 =sum2/count2
        Threshold =(1/2)*(A1+A2)
        
        c = Threshold2-Threshold
        print('Threshold new:',Threshold)
        
    ### step 7 ####
    for i in np.arange(height):
          for j in np.arange(width):
            a = I.item(i,j)
            if a > Threshold:           #Final Threshold
               b=255                    #initializing value G
               I.itemset((i,j) ,b)      # G(i,j) = 255 , if I(i,j) > T
            else:
               b=0                     #  G(i,j)= 0  , if I(i,j)  <=T
               I.itemset((i,j) ,b)
               
    plt.imshow(I, cmap=plt.get_cmap('gray'))     # final image plot
    plt.title("Resulting Image (Step 7)")
    plt.show()
   
    ####step 8-10 ####
   
    ##add Noise to image##.
    Icopy = I.astype('uint8')
    mean = 1.0   # some constant
    std = 1.0    # some constant (standard deviation)
    noisy_img = Icopy + np.random.normal(mean, std, Icopy.shape)
    I2 = np.clip(noisy_img, 0, 255)
    
    
    
    
    plt.imshow(I2, cmap=plt.get_cmap('gray'))
    plt.title(" Noise Addition (Step 8)")
    plt.show()
    
     #I need to convert the image to uint8 and scale the mask to 255. I create two#
    #structuring elements - one that is a 5 x 5 rectangle for the closing operation and #
    #another that is 2 x 2 for the opening operation. I run cv2.morphologyEx#
    #twice for the opening and closing operations respectively on the thresholded image.#
    
    #Once I do that,I multiply this mask with the#
    #original image so that we can grab the original pixels of the #
    #image back and maintaining what is considered a true object from the mask output.#
   
    #Remove spurious small islands of noise in an image#
    img_bw = 255*I2.astype('uint8')

    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    out = I2 * mask

    plt.imshow(out, cmap=plt.get_cmap('gray'))
    plt.title("Removal of Noise as small island removal (Step 9-10)")
    plt.show()
    
    ## median Filter ###
    out2 =cv2.medianBlur(I, 3)
    plt.imshow(out2, cmap=plt.get_cmap('gray'))
    plt.title("Removal of Noise median Filter (Step 9-10)")
    plt.show()
    