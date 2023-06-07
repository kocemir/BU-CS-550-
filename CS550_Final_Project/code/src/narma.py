import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def narma_generator():
    r=10
    total_time = 1380
    x_initial= np.array(np.random.uniform(low=0.0, high=0.5, size=(total_time,)))
    x_initial= x_initial.tolist()

    y_out = np.ones((r,))*0.1
    y_out=y_out.tolist()
    y_out[0]=0


    for yy in range(10,1380):

        y_out.append(y_out[yy-1]*0.3+ y_out[yy-1]*sum(y_out[yy-r:yy])*0.05+ 1.5*x_initial[yy-r]*x_initial[yy-1]+0.1)

    y_out= y_out[51:1331]
    y_out= np.array(y_out)
    y_out= y_out[np.newaxis,:]
    y_out= tf.convert_to_tensor(y_out)
    y_out=tf.cast(y_out,dtype="float32")

    
    return y_out