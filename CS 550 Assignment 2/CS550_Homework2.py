#!/usr/bin/env python
# coding: utf-8

# In[365]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[366]:


cur_dir = os.getcwd()
train1_path = os.path.join(cur_dir, 'train1.txt')
test1_path = os.path.join(cur_dir, 'test1.txt')
train2_path = os.path.join(cur_dir, 'train2.txt')
test2_path = os.path.join(cur_dir,'test2.txt')

train1_data = pd.read_csv(train1_path, delimiter='\t', names =['feature','label']).to_numpy()
test1_data = pd.read_csv(test1_path, delimiter='\t', names =['feature','label']).to_numpy()
train2_data = pd.read_csv(train2_path, delimiter='\t', names =['feature','label']).to_numpy()
test2_data = pd.read_csv(test2_path, delimiter='\t', names =['feature','label']).to_numpy()


# In[367]:


def Normalize(data):
    data_x= data[:,0]
    data_y= data[:,1]
    data[:,0]= (data_x-np.mean(data_x))/np.std(data_x) #(np.max(data_x)-np.min(data_x))
    data[:,1]= (data_y-np.mean(data_y))/np.std(data_y) #(np.max(data_y)-np.min(data_y))
    return data

#Normalized version of datasets
train1_norm= Normalize(train1_data)
test1_norm = Normalize(test1_data)
train2_norm = Normalize(train2_data)
test2_norm = Normalize(test2_data)


# In[368]:


# Data Exploration 
plt.subplot(2, 2, 1)
plt.scatter(train1_norm[:,0],train1_norm[:,1])
plt.title("Train Data 1")
plt.xlabel("Data point")
plt.ylabel("Label")

plt.subplot(2, 2, 2)
plt.scatter(test1_norm[:,0],test1_norm[:,1])
plt.title("Test Data 1")
plt.xlabel("Data point")
plt.ylabel("Label")

plt.subplot(2, 2, 3)
plt.scatter(train2_norm[:,0],train2_norm[:,1])
plt.title("Train Data 2")
plt.xlabel("Data point")
plt.ylabel("Label")

plt.subplot(2, 2, 4)
plt.scatter(test2_norm[:,0],test2_norm[:,1])
plt.title("Test Data 2")
plt.xlabel("Data point")
plt.ylabel("Label")

plt.suptitle("Normalized Data")
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1.0)


# In[369]:


# Train-Label Split
trainx1= train1_norm[:,0,np.newaxis]
trainx2= train2_norm[:,0,np.newaxis]
testx1= test1_norm[:,0,np.newaxis]
testx2= test2_norm[:,0,np.newaxis]


label_train1= train1_norm[:,1,np.newaxis].T
label_train2= train2_norm[:,1,np.newaxis].T
label_test1= test1_norm[:,1,np.newaxis].T
label_test2= test2_norm[:,1,np.newaxis].T


# In[370]:


def Initializer(init_choice="normal",shape=(1,1)):
    
    if init_choice== "normal":
        return np.random.normal(0.1,0.01,shape)
    elif init_choice== "uniform":
        return np.random.uniform(-0.1,0.1,shape)
    elif init_choice == "he":
        return np.random.normal(0.1,0.01,shape)*np.sqrt(2/shape[1])
    elif init_choice== "xavier":
        return np.random.normal(0.1,0.01,shape)*np.sqrt(2/(shape[0]+shape[1]))
    else:
        raise ValueError("Initialization Error")
      
    
def Forward_Activate(activ,data):
    
    if activ== "sigmoid":
        return  1/(1 + np.exp(-1*data))
    elif activ== "tanh":
        return np.tanh(data)
    elif activ=="ReLU":
        return data*(data>0)
    elif activ=="Linear":
        return data
    else:
        raise ValueError("Activation Error")
        
def Derivative_Activation(activ,data):
    
    if activ=="sigmoid":
        return Forward_Activate(activ,data)* (1-Forward_Activate(activ,data))
    elif activ=="tanh":
        return 1- (Forward_Activate(activ,data))**2
    
    elif activ=="ReLU":
        return 1*(data>0)
    elif activ=="Linear":
        return np.ones((data.shape))
    else:
        raise ValueError("Derivative Error")
        
    
    
def Loss(prediction,true_value,loss_choice="mse"):
    
    if loss_choice=="mse":
        sample_count = prediction.shape[1]
        total_error = np.sum((np.square(prediction-true_value)))
        mse= (1/(2*sample_count))*total_error
        return mse

    elif loss_choice=="binary_cross_entropy":

        total_error = np.sum(-(true_value*np.log(prediction)+(1-true_value)*np.log(1-prediction)))

        return (1/sample_count)*total_error
    else:
        
        raise TypeError("Choice of loss is wrong")


# In[381]:


class NeuralNetworkRegressor:
    
    def __init__(self,hidden_layers=0,hidden_units=None,activation="sigmoid",init_weight_choice= "uniform",loss="mse"):
        
        self.activation = activation
        self.init_weight_choice= init_weight_choice
        self.hidden_units= hidden_units
        self.hidden_layers= hidden_layers
        self.loss=loss
        
        
        if hidden_layers==0:
            self.W_input2output= Initializer(init_weight_choice, shape=(trainx1.shape[1],trainx1.shape[1]+1))
            
        else:
            self.W_input2hidden= Initializer(init_weight_choice, shape=(self.hidden_units,trainx1.shape[1]+1))
            self.W_hidden2output= Initializer(init_weight_choice, shape=(1,self.hidden_units+1))
            self.grad_cache_out= [0]
            self.grad_cache_in = [0]
    def FeedForward(self,data):
        sample_count = data.shape[0]
        
        # Add bias to input
        bias_term = np.ones((sample_count,1))
        self.input_modified= np.concatenate((bias_term,data), axis=1)        
        
        if self.hidden_layers==0:
            self.forward_prop=np.matmul(self.W_input2output,np.transpose(self.input_modified))
            output= Forward_Activate("Linear",self.forward_prop)
            return output
        else:
            
            self.hidden_activation = Forward_Activate(self.activation,np.matmul(self.W_input2hidden,np.transpose(self.input_modified)))
            self.hidden_activation_modified= np.concatenate((np.ones((1,sample_count)),self.hidden_activation),axis=0)
          
            output= Forward_Activate("Linear",np.matmul(self.W_hidden2output,self.hidden_activation_modified))
            return output
    
    def GradientCalculation(self,data,true_value):
        
        sample_count= data.shape[0]
        if self.hidden_layers==0:
            out= self.FeedForward(data)
          
            self.mse= Loss(out,true_value,self.loss)
            derivative_activation= Derivative_Activation("Linear",self.forward_prop)
            grad= (-1)*(1/sample_count)* np.matmul((true_value-out)*derivative_activation,self.input_modified)
            
            return grad, self.mse
        
        else:
            out= self.FeedForward(data)
            self.mse= Loss(out,true_value,self.loss)
            out_der = Derivative_Activation("Linear",np.matmul(self.W_hidden2output,self.hidden_activation_modified))
            out_delta= (true_value-out)*out_der
            grad_out=  (-1)*(1/sample_count)*np.matmul(out_delta,self.hidden_activation_modified.T)
            self.grad_cache_out.append(grad_out)
            
            hidden_nonactiv_der = Derivative_Activation(self.activation, np.matmul(self.W_input2hidden,np.transpose(self.input_modified)))
            delta_tilde_hidden = np.zeros((self.hidden_units,self.hidden_units))
            
            grad_input= np.zeros(self.W_input2hidden.shape)
            for  jj in range(hidden_nonactiv_der.shape[1]):
            
                for ii in range(self.hidden_units):
                    delta_tilde_hidden[ii,ii]= hidden_nonactiv_der[ii,jj]
                    
                out_delta_piece = out_delta[:,jj,np.newaxis]
                weight= np.transpose(self.W_hidden2output[:,1:])
                w_o = np.matmul(weight,out_delta_piece)
                del_hid= np.matmul(delta_tilde_hidden,w_o)
                grad_input+=np.matmul(del_hid,self.input_modified[np.newaxis,jj,:])
                
            grad_input= (-1)*(1/sample_count)*grad_input
            self.grad_cache_in.append(grad_input)
            return grad_input,grad_out,self.mse
            
             
             
     
    def BackPropagation(self,data,true_value, learning_rate=0.01,momentum=0,iteration=1):
        
        if self.hidden_layers==0:
            dW,loss= self.GradientCalculation(data,true_value)
            self.W_input2output -= (learning_rate*dW)
            
        else:
            dW_input,dW_out,loss= self.GradientCalculation(data,true_value)
            self.W_input2hidden -= (learning_rate*dW_input+momentum*self.grad_cache_in[iteration-1])
            self.W_hidden2output-= (learning_rate*dW_out+momentum*self.grad_cache_out[iteration-1])
                     
            
    def Train(self,epoch, data,true_value,lr=0.1,mt=0,update="batch"):
        
            if update=="batch":
                self.all_mse= []
                for ii in range(0,epoch):

                    self.BackPropagation(data,true_value,lr,mt)
                    self.all_mse.append(self.mse)
                    
            elif update=="sgd":
                
                self.all_mse= []
                iteration=0
                for ii in range(0,epoch):
                    
                    for dd in range(data.shape[0]):
                                iteration+=1
                                data_slice= data[np.newaxis,dd,:]
                                true_value_sliced= true_value[:,dd,np.newaxis]
                                self.BackPropagation(data_slice,true_value_sliced,lr,mt,iteration)

                                self.all_mse.append(self.mse)
    
    def Predict(self,data,label):
           pred= self.FeedForward(data)
           mean_squared_error=np.mean(np.square(label-pred))
           std_error = np.std(np.square(label-pred))
           return pred,mean_squared_error,std_error
            

def create_uniformed_data(input_data):
    
    sample_length = input_data.shape[0]
    y_max= np.max(input_data)
    y_min = np.min(input_data)
    uniformed_input = np.random.uniform(y_min,y_max,(sample_length*3,1))
    return uniformed_input


# # Part A

# # ****************  Dataset I *****************
# 

# In[433]:


# Training and Test Results 

NN_16= NeuralNetworkRegressor(1,16,"tanh","uniform","mse")
NN_16.Train(2000,trainx1,label_train1,0.01,0,"sgd")
prediction_training,mean_error,std_error = NN_16.Predict(trainx1,label_train1)
print("Mean error in training:", mean_error)
print("Standard deviation of error in training:", std_error)

prediction_test,mean_error,std_error = NN_16.Predict(testx1,label_test1)
print("Mean error in test:", mean_error)
print("Standard deviation of error in test:", std_error)


# In[416]:


# Create Uniformed Data
train_uniformed= create_uniformed_data(trainx1)
test_uniformed= create_uniformed_data(testx1)

train_uniformed_prediction,_,__= NN_16.Predict(train_uniformed,trainx1)
train_uniformed_prediction = train_uniformed_prediction .T

test_uniformed_prediction,_,__= NN_16.Predict(test_uniformed,testx1)
test_uniformed_prediction=test_uniformed_prediction.T

train_curve_data= list(sorted(zip(train_uniformed[:,0],train_uniformed_prediction[:,0])))
true_train, pred_train= zip(*train_curve_data)

test_curve_data= list(sorted(zip(test_uniformed[:,0],test_uniformed_prediction[:,0])))
true_test, pred_test= zip(*test_curve_data)


# In[417]:


# Training and Test Plots
plt.subplot(1,2,1)
plt.scatter(trainx1,label_train1,s=8)
plt.scatter(trainx1,prediction_training)
plt.plot(true_train,pred_train,c="red",lw=1.2)
plt.title("Training Results")
plt.xlabel("Data")
plt.ylabel("Value")
plt.legend(["Fitted Curve","Actual","Prediction"])

plt.subplot(1,2,2)
plt.scatter(testx1,label_test1,s=8)
plt.scatter(testx1,prediction_test)
plt.plot(true_test,pred_test,c="red",lw=1.2)
plt.title("Test Results")
plt.xlabel("Data")
plt.ylabel("Value")
plt.legend(["Fitted Curve","Actual","Prediction"])
plt.suptitle("Linear Regressor Dataset I")
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1.0)
plt.savefig("LinearR")


# # ************** Dataset II *****************

# In[476]:


NN_16= NeuralNetworkRegressor(1,8,"tanh","uniform","mse")
NN_16.Train(16000,trainx2,label_train2,0.01,0,"batch")
prediction_training,mean_error,std_error = NN_16.Predict(trainx2,label_train2)
print("Mean error in training:", mean_error)
print("Standard deviation of error in training:", std_error)

prediction_test,mean_error,std_error = NN_16.Predict(testx2,label_test2)
print("Mean error in test:", mean_error)
print("Standard deviation of error in test:", std_error)


# In[475]:


train_uniformed= create_uniformed_data(trainx2)
test_uniformed= create_uniformed_data(testx2)

train_uniformed_prediction,_,__= NN_16.Predict(train_uniformed,trainx2)
train_uniformed_prediction = train_uniformed_prediction .T

test_uniformed_prediction,_,__= NN_16.Predict(test_uniformed,testx2)
test_uniformed_prediction=test_uniformed_prediction.T

train_curve_data= list(sorted(zip(train_uniformed[:,0],train_uniformed_prediction[:,0])))
true_train, pred_train= zip(*train_curve_data)

test_curve_data= list(sorted(zip(test_uniformed[:,0],test_uniformed_prediction[:,0])))
true_test, pred_test= zip(*test_curve_data)

# Training and Test Plots
plt.subplot(1,2,1)
plt.scatter(trainx2,label_train2,s=8)
plt.scatter(trainx2,prediction_training)
plt.plot(true_train,pred_train,c="red",label="fitted curve",lw=1.2)
plt.title("Training Results")
plt.xlabel("Data")
plt.ylabel("Prediction")
plt.legend(["Fitter Curve","Actual","Prediction"],fontsize=7)

plt.subplot(1,2,2)
plt.scatter(testx2,label_test2,s=8)
plt.scatter(testx2,prediction_test)
plt.plot(true_test,pred_test,c="red",label="fitted curve",lw=1.2)
plt.title("Test Results")
plt.xlabel("Data")
plt.ylabel("Prediction")
plt.legend(["Fitter Curve","Actual","Prediction"],fontsize=7)
plt.suptitle("ANN with Learning Rate 0.01 and Batched Dataset II")
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1.0)
plt.savefig("mom02")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




