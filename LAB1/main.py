#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# %config Completer.use_jedi = False


# In[2]:


def generate_linear(n=100):
    import numpy as np
    pts = np.random.uniform(0, 1, (n, 2)) # 從(0,1)中透過uniform分配取出n個2維的點
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1] :
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1) # reshape完後變成row=100,clo=1的陣列

def generate_XOR_easy():
    import numpy as np
    inputs = []
    labels = []    

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i==0.5 :
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21,1)

def show_result(x,y,pred_y):
    import  matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.title('ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] ==0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

def sigmoid(x):
    return 1.0 / (1.0+np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0-x)

def relu (x):
    return np.maximum(0,x)

def derivative_relu(x):
    return x>0

# In[3]:


# Assume both hidden layer's width is 4

# x,y_ =generate_linear(n=100)
x,y_ =generate_XOR_easy()

class Model():
    def __init__(self, dims):
        self.weight = [
            np.random.uniform(size=(dims[i], dims[i+1]))
            for i in range(len(dims) - 1)]

# In[5]:

def training (itr, lr , width , input_x , label_y ):
    m = Model([2,width,width,1])

    y1 = np.zeros([y_.size,width])
    z1 = np.zeros([y_.size,width])
    y2 = np.zeros([y_.size,width])
    z2 = np.zeros([y_.size,width])
    y3 = np.zeros([y_.size,1])
    y  = np.zeros([y_.size,1])
    pred_y = np.zeros([y_.size,1])
        
    loss= np.zeros([itr,1])


    for epoch in range (itr):
    
        for i in range (y.size):

            # forward propagation
            y1[i] = x[i]  @ m.weight[0]
            z1[i] = sigmoid(y1[i])
            y2[i] = z1[i] @ m.weight[1]
            z2[i] = sigmoid(y2[i])
            y3[i] = z2[i] @ m.weight[2]
            y[i]  = sigmoid(y3[i])
            loss[epoch] = np.square(np.subtract(y, y_)).mean()
            # backward propagation
            dldy3  = 2*(y[i]-y_[i])  *derivative_sigmoid(y[i])
            dldw3  = z2[i] * dldy3

            dldy2  = (m.weight[2].T * dldy3) * derivative_sigmoid(z2[i])
            dldw2  = z1[i].reshape(width,1) @ dldy2

            dldy1  = (dldy2 @ m.weight[1].T) * derivative_sigmoid(z1[i])
            dldw1  = x[i].T.reshape(2,1) @ dldy1

            #update the weight
            m.weight[2] -= lr*dldw3.reshape(width,1)
            m.weight[1] -= lr*dldw2
            m.weight[0] -= lr*dldw1
        if(epoch%500==0):
            for i in range (y_.size):
                if(y[i]>0.5):pred_y[i] = 1
                else:pred_y[i]=0
            count = 0
            for i in range (y_.size):
                if(y_[i]==pred_y[i]):count+=1 
            print("epcoh =",epoch," loss =" , loss[epoch]," Acuuracy = ",count / y_.size)


    # In[8]:


    # plot the graph and accurracy
    show_result(x,y_,pred_y.reshape(y_.size))
    print("Acuuracy = " , count / y_.size)
    # plot the learning curve (loss, epoch curve)
    plt.plot(loss)
    plt.legend()
    plt.show()
if __name__ == '__main__':

    itr =5001
    lr=0.1
    width = 4
    plt.show()
    x,y_ =generate_linear(n=100)
    print("itr =", itr , " lr = " , lr , "in linear case")
    #x,y_ =generate_XOR_easy()
    #print("itr =", itr , " lr = " , lr , "width = " , width, "in XOR case")
    training(itr ,lr,width , x , y_)    

