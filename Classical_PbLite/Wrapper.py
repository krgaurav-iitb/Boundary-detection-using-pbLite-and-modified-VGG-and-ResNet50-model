#!/usr/bin/env python

"""

Author(s): 
Kumar Gaurav
"""

# Code starts here:

import numpy as np
import cv2
import scipy.signal as sg
from matplotlib import pyplot as plt
from scipy import ndimage
#rom np import linalg as la
import os
import math
import scipy
import scipy.stats as st
import skimage.transform
import sklearn.cluster
import argparse
def gaussKernel(sigx,sigy):
    
    bx=np.linspace(-10*sigx,10*sigx,40)
    by=np.linspace(-10*sigy,10*sigy,40)
    xx,yy= np.meshgrid(bx,by)
    kernel = np.exp(-0.5 * ((np.square(xx)/ np.square(sigx))+(np.square(yy) / np.square(sigy))))
    kernel/= kernel.max()
    kernel*=255.0
    
    return kernel

def FirstDerGaussian(sigx,sigy):
    
    bx=np.linspace(-10*sigx,10*sigx,40)
    by=np.linspace(-10*sigy,10*sigy,40)
    xx,yy= np.meshgrid(bx,by)
    kernel = np.exp(-0.5 * ((np.square(xx)/ np.square(sigx))+(np.square(yy) / np.square(sigy))))
    kernel=-1*xx*kernel
    kernel/= kernel.max()
    kernel*=255.0
    #print("fir",kernel.shape)
    return kernel

def SecondDerGaussian(sigx,sigy):
    
    bx=np.linspace(-10*sigx,10*sigx,40)
    by=np.linspace(-10*sigy,10*sigy,40)
    xx,yy= np.meshgrid(bx,by)
    kernel = np.exp(-0.5 * ((np.square(xx)/ np.square(sigx))+(np.square(yy) / np.square(sigy))))
    kernel=(-1+(np.square(xx)/ np.square(sigx)))*kernel
    kernel/= kernel.max()
    kernel*=255.0
    #print("sec",kernel.shape)
    return kernel

def Laplace(sigx,sigy):
    
    bx=np.linspace(-10*sigx,10*sigx,40)
    by=np.linspace(-10*sigy,10*sigy,40)
    xx,yy= np.meshgrid(bx,by)
    kernel = np.exp(-0.5 * ((np.square(xx)/ np.square(sigx))+(np.square(yy) / np.square(sigy))))
    kernel=(-1+((np.square(xx)/ np.square(sigx))+(np.square(yy) / np.square(sigy))))*kernel
    kernel/= kernel.max()
    kernel*=255.0
    #print("sec",kernel.shape)
    return kernel

def sobel_filters( kernel):
    
    a1 = np.matrix([1, 2, 1])
    a2 = np.matrix([1, 0, -1])
    SoBx = a1.T * a2
    #convolution with gaussian kernel
    convx= sg.convolve2d(kernel,SoBx,boundary='symm', mode='same')
    convx/=convx.max()
    convx*=255.0
    return convx

def gabor(sigma, theta, Lambda, psi, gamma):
    
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    gb/=gb.max()
    gb*=255.0
    return gb
def brightnessMap(Img, NumberOfClusters):
    p, q = Img.shape
    Input = np.reshape(Img, ((p * q), 1))
    kmeans = sklearn.cluster.KMeans(n_clusters = NumberOfClusters, random_state = 2)
    kmeans.fit(Input)
    labels = kmeans.predict(Input)
    result = np.reshape(labels, (p, q))
    #plt.imshow(result, cmap = 'binary')
    return result

def colorMap(Img, NumberOfClusters):
    p,q,r = Img.shape
    Input = np.reshape(Img,((p * q), r))
    kmeans = sklearn.cluster.KMeans(n_clusters = NumberOfClusters, random_state = 2)
    kmeans.fit(Input)
    labels = kmeans.predict(Input)
    result = np.reshape(labels, (p, q))
    #plt.imshow(result)
    return result

def generateHalfDiskMasks(NumberOfScales, NumberOfOrients):
    result = list()
    Orientations = np.linspace(0,360,NumberOfOrients)

    for i in NumberOfScales:
        g = list()
        a, b = generateHalfDisks(Radius = i)
        for i, orient in enumerate(Orientations):
            r1 = skimage.transform.rotate(b, orient, cval =1)
            o1 = np.logical_or(a, r1)
            o1 = o1.astype(np.int)
            b2 = np.flip(b, 1)
            r2 = skimage.transform.rotate(b2, orient, cval =1)
            o2 = np.logical_or(a, r2)
            o2 = o2.astype(np.int)
            result.append(o1)
            result.append(o2)

    
    return result
def generateHalfDisks(Radius):
    a = np.ones((2 * Radius + 1, 2 * Radius + 1))
    y, x = np.ogrid[-Radius:Radius + 1, -Radius:Radius + 1]
    mask = x*x + y*y <= Radius**2
    a[mask] = 0
    b = np.ones((2 * Radius + 1, 2 * Radius + 1))
    y,x = np.ogrid[-Radius:Radius + 1, -Radius:Radius + 1]
    p = x > -1
    q = y > -Radius - 1
    mask2 = p * q
    b[mask2] = 0
    return a, b
def calculateChiSquareGradient(Img, Bins, filter1, filter2):
    ChiSquareDistance = Img * 0
    g = list()
    h = list()
    for i in range(Bins):
        img = np.ma.masked_where(Img == i, Img)
        img = img.mask.astype(np.int)
        g = cv2.filter2D(img, -1, filter1)
        h = cv2.filter2D(img, -1, filter2)
        if (g + h).all() != 0:
            ChiSquareDistance = ChiSquareDistance + ((g - h)**2 /(g + h))
    return ChiSquareDistance/2
def gradient(Img, Bins, FilterBank):
    Grad = Img
    for n in range(len(FilterBank)/2):
        ChiGrad = calculateChiSquareGradient(Img, Bins, FilterBank[2 * n], FilterBank[2 * n + 1])
        Grad = np.dstack((Grad, ChiGrad))
    result = np.mean(Grad, axis = 2)
    return result
    
def main():

        """
        Generate Difference of Gaussian Filter Bank: (DoG)
        Display all the filters in this filter bank and save image as DoG.png,
        use command "cv2.imwrite(...)"
        """
        
        DOG_FB=[]
        for j in range(2):
            #sigma=np.array([[((j*3)+1), 0],[0, ((j*3)+1)]])
            #sigma=1.0
            kernel=gaussKernel(((j*3)+1), ((j*3)+1))
            img=sobel_filters( kernel)
            path="/home/kumar/Desktop/MSCS/AdvanceCV/DOG"
            for i in range(16):
                rot_img = ndimage.rotate(img, i*22.5, reshape=True)
                DOG_FB.append(rot_img)
                cv2.imwrite(os.path.join(path,'DoG'+str(j)+'-'+str(i)+'.png'), rot_img)
        
        """
        
        Generate LM filter
        
        """
        
        sig_scales_small=[1.0,np.sqrt(2),2.0]
        sig_scales_large=[1.0,np.sqrt(2),2.0,2*np.sqrt(2),4.0,4*np.sqrt(2),8.0,8*np.sqrt(2)]
        
        LM=[]
        for j in sig_scales_small:
            #sigma=np.array([[j, 0],[0,(3*j)]])
            #kernel=gaussKernel(j,3*j)
            img=FirstDerGaussian(j,3*j)
            path1="/home/kumar/Desktop/MSCS/AdvanceCV/LM/1"
            for i in range(6):
                rot_img = ndimage.rotate(img, i*60, reshape=True)
                LM.append(rot_img)
                cv2.imwrite(os.path.join(path1,'LMs'+str(round(j,2))+'-'+str(round(i,2))+'.png'), rot_img)        
        
        for j in sig_scales_small:
            #sigma=np.array([[j, 0],[0,(3*j)]])
            #kernel=gaussKernel(j,3j)
            img=SecondDerGaussian(j,3*j)
            path2="/home/kumar/Desktop/MSCS/AdvanceCV/LM/2"
            for i in range(6):
                rot_img = ndimage.rotate(img, i*60, reshape=True)
                LM.append(rot_img)
                cv2.imwrite(os.path.join(path2,'LMD'+str(round(j,2))+'-'+str(round(i,2))+'.png'), rot_img)        
    
    
        for j in range(8):
           #k1=gaussKernel(sig_scales_large[j],sig_scales_large[j])
           #k2=gaussKernel(np.sqrt(2)*sig_scales_large[j],np.sqrt(2)*sig_scales_large[j])
           #LoG=k2-k1
           LoG=Laplace(sig_scales_large[j],3*sig_scales_large[j])
           LoG/=LoG.max()
           LoG*=255.0
           LM.append(LoG)
           path2="/home/kumar/Desktop/MSCS/AdvanceCV/LoG"
           cv2.imwrite(os.path.join(path2,'LOG'+str(round(j,2))+'.png'), LoG)
    
        for j in range(4):
           k=gaussKernel(sig_scales_large[j],sig_scales_large[j])
           LM.append(k)
           path2="/home/kumar/Desktop/MSCS/AdvanceCV/LoG"
           cv2.imwrite(os.path.join(path2,'Gaussian'+str(round(j,2))+'.png'), k)

        """
        
        Generate Gabor filter
        
        """
        Gab=[]
        for j in sig_scales_large[0:5]:
            #sigma=np.array([[j, 0],[0,(3*j)]])
            #kernel=gaussKernel(j,3*j)
            img=FirstDerGaussian(j,3*j)
            path1="/home/kumar/Desktop/MSCS/AdvanceCV/gabor"
            for i in range(8):
                rot_img = gabor(j,45*i,1.0,1.0,1.0)
                Gab.append(rot_img)
                cv2.imwrite(os.path.join(path1,'Gab'+str(round(j,2))+'-'+str(round(i,2))+'.png'), rot_img) 
                
        """Textonmap"""

        #filterBank={DOG_FB,LM,Gab}
        
        path3="/home/kumar/Desktop/MSCS/AdvanceCV/YourDirectoryID_hw0/Phase1/BSDS500/Images/"
        for i in range(1,11):
            #path3+=str(i)+".jpg"
            #print(path)
            Im=np.float32(cv2.imread(os.path.join(path3,str(i)+".jpg")))
            Im1=np.float32(cv2.imread(os.path.join(path3,str(i)+".jpg"),0))
            Fil_Im=np.array(Im1)
            print("yaY",i)
            for j in range(32):
                #print(Im)
                a=cv2.filter2D(Im1, -1, DOG_FB[j])
                Fil_Im=np.dstack((Fil_Im,a))
            for j in range(48):
                Fil_Im=np.dstack((Fil_Im,cv2.filter2D(Im1, -1, LM[j])))
            for j in range(40):
                Fil_Im=np.dstack((Fil_Im,cv2.filter2D(Im1, -1, Gab[j])))
                
            p, q, _ = Im.shape
            TextonMap = Fil_Im[:,:,1:]
            x, y, z = TextonMap.shape
            Input = np.reshape(TextonMap, ((p * q), z))
            kmeans = sklearn.cluster.KMeans(n_clusters = 64, random_state = 2)
            kmeans.fit(Input)
            labels = kmeans.predict(Input)
            T = np.reshape(labels,(x, y))
            plt.imsave(str(i) + "_TextonMap" + ".png", T) 
            
            """BrightnessMap"""
            B = brightnessMap(Im1, 16)
            plt.imsave(str(i) + "_BrightMap" + ".png", B)
            
            """ColorMap"""
            C = colorMap(Im, 16)
            plt.imsave(str(i) + "_ColorMap" + ".png", C)
            
            """
            Generate Half-disk masks
            Display all the Half-disk masks and save image as HDMasks.png,
            use command "cv2.imwrite(...)"
            """
            c = generateHalfDiskMasks([5,7,16], 8)
    
            """
            Generate Texton Gradient (Tg)
            Perform Chi-square calculation on Texton Map
            Display Tg and save image as Tg_ImageName.png,
            use command "cv2.imwrite(...)"
            """
            Tg = gradient(T, 64, c)
            plt.imsave("Tg_Image" + str(i) + ".png", Tg)
    
            """
            Generate Brightness Gradient (Bg)
            Perform Chi-square calculation on Brightness Map
            Display Bg and save image as Bg_ImageName.png,
            use command "cv2.imwrite(...)"
            """
            Bg = gradient(B, 16, c)
            plt.imsave("Bg_Image" + str(i) + ".png", Bg, cmap = 'binary')
    
            """
            Generate Color Gradient (Cg)
            Perform Chi-square calculation on Color Map
            Display Cg and save image as Cg_ImageName.png,
            use command "cv2.imwrite(...)"
            """
            Cg = gradient(C, 16, c)
            plt.imsave("Cg_Image" + str(i) + ".png", Cg)
    
            """
            Read Sobel Baseline
            use command "cv2.imread(...)"
            """
            sobelBaseline = plt.imread('/home/kumar/Desktop/MSCS/AdvanceCV/YourDirectoryID_hw0/Phase1/BSDS500/SobelBaseline/'+str(i)+'.png',0)
    
            """
            Read Canny Baseline
            use command "cv2.imread(...)"
            """
            cannyBaseline = plt.imread('/home/kumar/Desktop/MSCS/AdvanceCV/YourDirectoryID_hw0/Phase1/BSDS500/CannyBaseline/'+str(i)+'.png',0)
    
            """
            Combine responses to get pb-lite output
            Display PbLite and save image as PbLite_ImageName.png
            use command "cv2.imwrite(...)" 
            """
            temp = (Tg + Bg + Cg) / 3
            PbLiteOutput = np.multiply(temp, (0.1 * cannyBaseline + 0.9 * sobelBaseline))
            cv2.imwrite("PbLite_Image" + str(i) + ".png", PbLiteOutput)

            plt.imshow(PbLiteOutput, cmap = 'binary')
if __name__ == '__main__':
    main()
 


