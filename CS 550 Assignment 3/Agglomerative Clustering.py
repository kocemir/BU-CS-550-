#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time
from scipy.spatial import distance


# In[58]:


# Read Image from its directory
cur_dir = os.getcwd()
image_path = os.path.join(cur_dir, 'sample.jpg')
sample_image = mpimg.imread(image_path)
plt.imshow(sample_image)


# In[59]:


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
    plt.show()
    
    
def ImagePlot(k_list,image_list):
    
    plt.figure(figsize=(8,8))
    for ii in range(len(k_list)):
     
        plt.subplot(3,2,ii+1)
        plt.imshow(image_list[ii])
        plt.title('K={}'.format((k_list[ii])))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5 )
        
            
        


# In[178]:


class AgglomerativeClustering():
    
    def __init__(self,image,K_initial, K_final):
    
        self.K_initial=K_initial
        self.K_final= K_final
     
        self.similarity_matrix=0
        self.cluster_number = 0
        self.cluster_list = 0
        
        km=KMeansCluster(image,K_initial)
        init_duration, init_error = km.Train(0.0001,False)
        self.data=km.data
        
        self.mean= km.mean
        self.std=km.std
        
        self.centroid=km.centroid
        self.membership=km.assign2cluster
    
    def DistanceCalculation(self,cluster_1,cluster_2,linkage):
        
        
        dist_matrix= distance.cdist(cluster_1,cluster_2,"euclidean")
        if linkage=="single":
            return np.min(dist_matrix)
        elif linkage=="complete":
            return np.max(dist_matrix)
        else:
             raise ValueError('Linkage type is not in the list')
            
    def DistanceMatrix(self,linkage):
        distance_matrix = np.zeros((self.cluster_number,self.cluster_number))
        start= time.time()
        print("Started to form distance matrix")
        counter=0
        for ii in range(self.cluster_number):
         
            for jj in range(self.cluster_number):
                counter=counter+1
                print(counter)
                if ( ii!=jj and distance_matrix[ii,jj]==0 and distance_matrix[jj,ii]==0):
                    
                    dist= self.DistanceCalculation(self.data[self.membership==jj], self.data[self.membership==ii],linkage)
                    distance_matrix[ii,jj ]=dist
                    distance_matrix[jj,ii]=dist
                
                elif distance_matrix[ii,jj]!=0 :
                    distance_matrix[ii,jj] = distance_matrix[jj,ii]
            
                elif distance_matrix[jj,ii]!=0:
                    distance_matrix[jj,ii] = distance_matrix[ii,jj]
        end= time.time()
        self.similarity_matrix=distance_matrix
        
        print("Distance matrix is created in",end-start,"seconds.")
        return end-start
    def Update(self,linkage):
        
      
        indices= np.argwhere(self.similarity_matrix == np.amin(self.similarity_matrix[self.similarity_matrix!=0]))[0]
        ind1= indices[0]
        ind2= indices[1]
        
        self.membership[self.membership==self.cluster_list[ind2]]=self.cluster_list[ind1]
        
        if linkage=="single":
            self.similarity_matrix[ind1,:] = np.minimum(self.similarity_matrix[ind2,:],self.similarity_matrix[ind1,:])
            self.similarity_matrix[:,ind1] = np.minimum(self.similarity_matrix[:,ind2],self.similarity_matrix[:,ind1])
            self.similarity_matrix[ind1,ind1] =0
        elif linkage=="complete":
            self.similarity_matrix[ind1,:] = np.maximum(self.similarity_matrix[ind2,:],self.similarity_matrix[ind1,:])
            self.similarity_matrix[:,ind1] = np.maximum(self.similarity_matrix[:,ind2],self.similarity_matrix[:,ind1])
            self.similarity_matrix[ind1,ind1] =0
        self.similarity_matrix= np.delete(self.similarity_matrix,ind2,axis=0)
        self.similarity_matrix= np.delete(self.similarity_matrix,ind2,axis=1)
        
        
        removal = self.cluster_list[ind2]
        self.cluster_number-=1
        self.cluster_list=self.cluster_list[self.cluster_list!=self.cluster_list[ind2]]
    
        return removal
    
    def Train(self,linkage,err_dif=0.001,verbose=False):
        
        print("Start to train!")
        self.cluster_number = self.K_initial
        self.cluster_list = np.unique(self.membership)
        matrix_time= self.DistanceMatrix(linkage)
        
        start=time.time()
        while(self.cluster_number > self.K_final):
            rem = self.Update(linkage)
            if verbose:
                print('%d merged, %d clusters remain' % (rem, self.cluster_number))
        end=time.time()
      
        print("Ended training and last:",end-start+matrix_time,"seconds.")
        return end-start+matrix_time
    def Centroid(self):
        
        for ii, cc in enumerate(self.cluster_list):
            self.centroid[ii,:]=np.mean(self.data[self.membership==cc],axis=0)
    def TotalError(self):
        
        total_error=0
        for ee,cc in enumerate(self.cluster_list):
            cluster_error = np.sum(np.sqrt(np.sum((self.data[self.membership==cc]-self.centroid[ee,:])**2,axis=1)))
            total_error+=cluster_error
        return total_error/(self.data.shape[0])
                                   
                                   
    def DenormalizeImage(self):
           
            self.clustered_data= np.ones(self.data.shape)
            for cc, cl in enumerate(self.cluster_list):
                self.clustered_data[self.membership==cl]= (self.centroid[np.newaxis,cc,:]*self.std)+self.mean
            
            self.clustered_image = np.round(self.clustered_data.reshape(sample_image.shape)).astype(int)
            return self.clustered_image
        
                                   
                                   
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[186]:


#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time
from scipy.spatial import distance


# In[58]:


# Read Image from its directory
cur_dir = os.getcwd()
image_path = os.path.join(cur_dir, 'sample.jpg')
sample_image = mpimg.imread(image_path)
plt.imshow(sample_image)


# In[59]:


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
    plt.show()
    
    
def ImagePlot(k_list,image_list):
    
    plt.figure(figsize=(8,8))
    for ii in range(len(k_list)):
     
        plt.subplot(3,2,ii+1)
        plt.imshow(image_list[ii])
        plt.title('K={}'.format((k_list[ii])))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5 )
        
            
        


# In[122]:


class AgglomerativeClustering():
    
    def __init__(self,image,K_initial, K_final):
    
        self.K_initial=K_initial
        self.K_final= K_final
     
        self.similarity_matrix=0
        self.cluster_number = 0
        self.cluster_list = 0
        
        km=KMeansCluster(image,K_initial)
        init_duration, init_error = km.Train(0.00000001,False)
        self.data=km.data
        
        self.mean= km.mean
        self.std=km.std
        
        self.centroid=km.centroid
        self.membership=km.assign2cluster
    
    def DistanceCalculation(self,cluster_1,cluster_2,linkage):
        
        
        dist_matrix= distance.cdist(cluster_1,cluster_2,"euclidean")
        if linkage=="single":
           
                   return np.min(dist_matrix)
        elif linkage=="complete":
            
                  return np.max(dist_matrix)
        else:
             raise ValueError('Linkage type is not in the list')
            
    def DistanceMatrix(self,linkage):
        distance_matrix = np.zeros((self.cluster_number,self.cluster_number))
        start= time.time()
        print("Started to form distance matrix")
        counter=0
        for ii in range(self.cluster_number):
         
            for jj in range(self.cluster_number):
                counter=counter+1
                print(counter)
                if ( ii!=jj and distance_matrix[ii,jj]==0 and distance_matrix[jj,ii]==0):
                    
                    dist= self.DistanceCalculation(self.data[self.membership==ii], self.data[self.membership==jj],linkage)
                    distance_matrix[ii,jj ]=dist
                    distance_matrix[jj,ii]=dist
                
                elif distance_matrix[ii,jj]!=0 :
                    distance_matrix[ii,jj] = distance_matrix[jj,ii]
            
                elif distance_matrix[jj,ii]!=0:
                    distance_matrix[jj,ii] = distance_matrix[ii,jj]
        end= time.time()
        self.similarity_matrix=distance_matrix
        
        print("Distance matrix is created in",end-start,"seconds.")
        return end-start
    def Update(self,linkage):
        
      
        indices= np.argwhere(self.similarity_matrix == np.amin(self.similarity_matrix[self.similarity_matrix!=0]))[0]
        ind1= indices[0]
        ind2= indices[1]
        
        self.membership[self.membership==self.cluster_list[ind2]]=self.cluster_list[ind1]
        
        if linkage=="single":
            self.similarity_matrix[ind1,:] = np.minimum(self.similarity_matrix[ind2,:],self.similarity_matrix[ind1,:])
            self.similarity_matrix[:,ind1] = np.minimum(self.similarity_matrix[:,ind2],self.similarity_matrix[:,ind1])
            self.similarity_matrix[ind1,ind1] =0
        elif linkage=="complete":
            self.similarity_matrix[ind1,:] = np.maximum(self.similarity_matrix[ind2,:],self.similarity_matrix[ind1,:])
            self.similarity_matrix[:,ind1] = np.maximum(self.similarity_matrix[:,ind2],self.similarity_matrix[:,ind1])
            self.similarity_matrix[ind1,ind1] =0
        self.similarity_matrix= np.delete(self.similarity_matrix,ind2,axis=0)
        self.similarity_matrix= np.delete(self.similarity_matrix,ind2,axis=1)
        
        
        removal = self.cluster_list[ind2]
        self.cluster_number-=1
        self.cluster_list=self.cluster_list[self.cluster_list!=self.cluster_list[ind2]]
    
        return removal
    
    def Train(self,linkage,err_dif=0.001,verbose=False):
        
        print("Start to train!")
        self.cluster_number = self.K_initial
        self.cluster_list = np.unique(self.membership)
        matrix_time=self.DistanceMatrix(linkage)
        
        start=time.time()
        while(self.cluster_number > self.K_final):
            rem = self.Update(linkage)
            if verbose:
                print('%d merged, %d clusters remain' % (rem, self.cluster_number))
        end=time.time()
      
        print("Ended training and last:",end-start+matrix_time,"seconds.")
        return end-start+matrix_time
    def Centroid(self):
        
        for ii, cc in enumerate(self.cluster_list):
            self.centroid[ii,:]=np.mean(self.data[self.membership==cc],axis=0)
    def TotalError(self):
        
        total_error=0
        for ee,cc in enumerate(self.cluster_list):
            cluster_error = np.sum(np.sqrt(np.sum((self.data[self.membership==cc]-self.centroid[ee,:])**2,axis=1)))
            total_error+=cluster_error
        print("Total error:", total_error/(self.data.shape[0]))
        return total_error/(self.data.shape[0])
                                   
                                   
    def DenormalizeImage(self):
           
            self.clustered_data= np.ones(self.data.shape)
            for cc,cl in enumerate(self.cluster_list):
                self.clustered_data[self.membership==cl]= (self.centroid[np.newaxis,cc,:]*self.std)+self.mean
            
            self.clustered_image = np.round(self.clustered_data.reshape(sample_image.shape)).astype(int)
            return self.clustered_image
        
                                   
                                   
    


# In[123]:

ag= AgglomerativeClustering(sample_image,100,60)

ag.Train("single",0.000001,True)
err=ag.TotalError()
image=ag.DenormalizeImage()
plt.title("Single Linkage K=30")
plt.imsave("single_image.jpg",np.uint8(image))


ag= AgglomerativeClustering(sample_image,100,60)

ag.Train("complete",0.000001,True)
err=ag.TotalError()
image=ag.DenormalizeImage()
plt.title("Complete Linkage K=30")
plt.imsave("complete_image.jpg",np.uint8(image))


print("STARTED")
total_error= []
k_list = [2,4,6,8,10,30,40,60,80]

run_time_list = []

for kk in k_list:
   ag= AgglomerativeClustering(sample_image,100,kk)
   duration= ag.Train("complete",0.000001,True)
   err=ag.TotalError()
   image=ag.DenormalizeImage()
   image_title= "clustered_k"+str(kk)+".jpg"
   plt.title("K= "+str(kk))
   plt.imsave(image_title,np.uint8(image))
   total_error.append(err)
   run_time_list.append(duration)
   print("******************* DONE *************************** ", kk ) 

plt.plot(k_list,total_error)
plt.xlabel("K")
plt.ylabel("MSE")
plt.savefig("KvsMSE.jpg")
   
with open('run_time.txt', 'w') as f:
    for item in run_time_list:
        f.write("%s\n" % item)


# In[187]:


total_error


# In[189]:


plt.plot(k_list,total_error)

plt.xlabel("K")
plt.ylabel("MSE")
plt.savefig("KvsMSE.jpg")
   


# In[202]:


plt.figure(figsize=(8,8))
for ii in range(len(k_list)):
    kk=k_list[ii]
    title="clustered_"+"k"+str(kk)+".jpg"
    img= mpimg.imread(title)

    plt.subplot(3,3,ii+1)
    plt.title("K= "+ str(kk))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)
    plt.imshow(img)
plt.savefig("KvsImage.jpg")


# In[ ]:




