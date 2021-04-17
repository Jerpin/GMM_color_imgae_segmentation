#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:18:51 2021

@author: jerry
"""

from sklearn.mixture import GaussianMixture
from skimage import io
import statistics

"""
image1: Fit GMM model
image2: Predict target
image_ans2: the answer of image2
gaussians:gaussians requires (not larger than 6)
For gaussians, user can change the maximum number of gaussians at line 69 by adding elif
(See line 55)
"""
def accuracy_GMM(image1 , image2 , image_ans , gaussians):#max of gaussian = 6
    #image1 = io.imread('/Users/jerry/Downloads/hw2/soccer1.jpg')
    #flatten image1 & 2
    flatten1 = []
    flatten2 = []
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            flatten1.append(image1[i][j])
    for i in range(image2.shape[0]):
        for j in range(image2.shape[1]):
            flatten2.append(image2[i][j])    

    #fit image1
    n_gaussians = gaussians
    M1 = GaussianMixture(n_components=n_gaussians, random_state=0).fit(flatten1)
    #pridict every pixel in image2
    p1 = M1.predict(flatten2)

    #Flatten the answer image and chang it in to binary
    flatten_ans = []
    for i in range(image_ans.shape[0]):
        for j in range(image_ans.shape[1]):
            if image_ans[i][j][0] == 255:
                binary = 1
            elif image_ans[i][j][0] == 0:
                binary = 0
            flatten_ans.append(binary)
            
    #Generate empy list to save the numbers of pixels belongs to different gaissians
    g = []
    for i in range(n_gaussians):
        g.append([])
    
    #Condiser each pixel that belongs to different Gaussian
    #For different Gaussian, add the answer to g[] corresponding to the Gaussian
    #Add "elif" if you want to have more guassians    
    for i in range(len(p1)):
        if p1[i] == 0:
            g[0].append(flatten_ans[i])
        elif p1[i] == 1:
            g[1].append(flatten_ans[i])
        elif p1[i] == 2:
            g[2].append(flatten_ans[i])
        elif p1[i] == 3:
            g[3].append(flatten_ans[i])
        elif p1[i] == 4:
            g[4].append(flatten_ans[i])
        elif p1[i] == 5:
            g[5].append(flatten_ans[i])
            
    #For different Gaussian, measure the average  
    #If the average is higher than 0.5, we consider this Gaussian is correct       
    g_considered_correct = []       
    for i in range(len(g)):
        if statistics.mean(g[i])>0.5:
            g_considered_correct.append(i)
    
    #'1' represents pixels belong to Gaussians considered correct
    #Here, we generate a flattened binary image in order to compare with the answer
    con_correct = []
    count_correct = 0
    for i in range(len(p1)):
        if p1[i] in g_considered_correct:
            con_correct.append(1)
        else:
            con_correct.append(0)
        # determine if it is correct(1==1,0==0)
        if con_correct[i] == flatten_ans[i]:
            count_correct +=1
    
    #Measure the accuracy
    accuracy = count_correct/len(flatten_ans)
    print('M1:image2 accuracy( %s Gaussions): %.5f'  %(n_gaussians,accuracy))

image1 = io.imread(input('url(image1(training)):'))
image2 = io.imread(input('url(image2(testing)):'))
image_ans = io.imread(input('url(image_ans):'))
gaussians = int(input('Number of Gaussians(no larger than 6):'))
accuracy_GMM(image1,image2 , image_ans , gaussians)
