#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time


# In[32]:


# Read Image from its directory
cur_dir = os.getcwd()
image_path = os.path.join(cur_dir, 'sample.jpg')
sample_image = mpimg.imread(image_path)
plt.imshow(sample_image)


# In[36]:


class KMeansCluster():
    
    def __init__(self,image,K=2):
        
        self.rows,self.cols,self.chans= image.shape
        
        self.data= image.reshape((self.rows*self.cols,self.chans))
        self.cluster_number= K
        
                
        permuted = np.random.permutation(self.rows*self.cols)
        K_ind= permuted[0:self.cluster_number]
        self.centroid = np.array(self.data[K_ind,:])
       
        
        # You can switch it off if you do not standardize 
        self.Normalizer()
        
    def Normalizer(self):
        
        mean= np.mean(self.data,axis=0)
        mean=mean[np.newaxis]
        self.mean=mean
        stn= np.std(self.data,axis=0)
        stn= stn[np.newaxis]
        self.std= stn
        self.data = (self.data-mean)/stn
        
            
        permuted = np.random.permutation(self.rows*self.cols)
        K_ind= permuted[0:self.cluster_number]
        self.centroid = np.array(self.data[K_ind,:])
       
        
    
    def Expectation_Maximization(self):
        
        
        total_error=0
        distance2center=  np.sqrt(np.sum((self.data-self.centroid[:,np.newaxis,:])**2,axis=-1))
        #print(distance2center.shape)
        distance2center= distance2center.T
        self.assign2cluster= np.argmin(distance2center,axis=1)
        
       # print(self.centroid[:,np.newaxis].shape)
       # print(distance2center.shape)
      
        
        
        # Maximization Part
   
        for cl in range(self.cluster_number):
            
            cluster_points = self.data[self.assign2cluster==cl]
          
            self.centroid[cl,:]=np.mean(cluster_points,axis=0)
            #print(self.centroid[cl,:])
            cluster_error =  np.sum(np.sum(np.sqrt((cluster_points-self.centroid[cl,:])**2),axis=-1),axis=0)
            total_error+=cluster_error
        mean_error = total_error/self.data.shape[0]
       
        return mean_error
    
    def Train(self,stop_criterion=0.0001,verbose=True):
        error_list = [np.inf]
        flag = True
        start= time.time()
        while(flag):
             
             if verbose==True:
                print("Current total error is:", error_list[-1])
             current_error=self.Expectation_Maximization()
             diff=  np.abs(current_error-error_list[-1])
             if diff <= stop_criterion:
                    flag=False
             error_list.append(current_error)
            
        end= time.time() 
        if verbose==True:
            print("Traning duration is", end-start,"seconds")
        return end-start,current_error
    
    def DenormalizeImage(self):
        
            self.clustered_data= np.ones(self.data.shape)
            for cl in range(self.cluster_number):
                self.clustered_data[self.assign2cluster==cl]= (self.centroid[np.newaxis,cl,:]*self.std)+self.mean
            
            self.clustered_image = np.round(self.clustered_data.reshape(self.rows,self.cols,self.chans)).astype(int)
            return self.clustered_image
    
    

def BestChooseK(K_List,data):
    error_list= []
    image_list= []
    time_list = []
    
    for kk in K_List:
        print("Started to run on K:", kk)

        
        km= KMeansCluster(data,kk)
        tr,error=km.Train(verbose=False)
        error_list.append(error)
        time_list.append(tr)
       
        im = km.DenormalizeImage()
        image_list.append(im)
        print("Ended to run on K:",kk)
        print("Run time is ",time_list[-1]," seconds." )
        print("Error is ", error_list[-1])
        print("****************************************************")
    return error_list,time_list,image_list
                
                
def ErrorPlot(k_list,error_list):
    
    plt.plot(k_list,error_list)
    plt.xlabel("K")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE vs K")
    plt.show()
    
    
def ImagePlot(k_list,image_list):
    
    plt.figure(figsize=(8,8))
    for ii in range(len(k_list)):
     
        plt.subplot(3,2,ii+1)
        plt.imshow(image_list[ii])
        plt.title('K={}'.format((k_list[ii])))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5 )
        
            
        


# In[34]:


k_list = [2,3,4,5,6,15,25,35,45,90,120]
error_list,time_list,image_list =  BestChooseK(k_list,sample_image)


# In[37]:



ErrorPlot(k_list,error_list)


# In[22]:


k_image_list = [2,3,4,5,6,15]
ImagePlot(k_image_list,image_list)


# ##  DONE!
