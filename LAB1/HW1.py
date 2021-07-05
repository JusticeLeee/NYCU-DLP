'''
DLP-Lab1 back-propagation
'''

import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):

    pts = np.random.uniform(0, 1, (n, 2)) # low bound, high bound, size
    #print(pts)
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1] :
            labels.append(0)
        else:
            labels.append(1)
    #print(labels)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():

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

# nn forward and backward functions
def forward_product(x, w, b):
    return np.matmul(x, w)# +b I didn't use bias in this lab

def backward_product(dLdy, x, w):
    dLdx = np.matmul(dLdy, w.transpose())
    dLdw = np.matmul(x.transpose(), dLdy)
    dLdb = 0 #dLdy 
    return dLdx, dLdw, dLdb
  
def forward_sigmoid(x):
    return 1.0/(1.0 +np.exp(-x))
def backward_sigmoid(x): # derivative of sigmoid
    return np.exp(-x) / np.square( 1+np.exp(-x) )

def forward_ReLU(x):
    return x * (x>0)
def backward_ReLU(x): # derivative of ReLU
    return (x>0) 

class AdamOptim():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
    def update(self, t, w, b, dw, db):
        ## dw, db are from current minibatch
        ## momentum beta 1
        # * weights * #
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        # * biases * #
        self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

        ## rms beta 2
        # * weights * #
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        # * biases * #
        self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db)

        ## bias correction
        m_dw_corr = self.m_dw/(1-self.beta1**t)
        m_db_corr = self.m_db/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)
        v_db_corr = self.v_db/(1-self.beta2**t)

        ## update weights and biases
        w = w + self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        b = b + self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
        return w, b


class simple_NN:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.optimizers = {}
    
    def build_NN(self, num_of_node):

        # input layer weight
        self.input_layer_w = np.random.uniform(-0.1, 0.1, (self.input_size, num_of_node))
        self.input_layer_b = np.random.uniform(0, 0, (1, num_of_node))     
        # self.optimizers["input_layer"] = AdamOptim() # not used

        # hidden layer weight
        self.hidden_layer_w = np.random.uniform(-0.1, 0.1, (num_of_node, num_of_node))
        self.hidden_layer_b = np.random.uniform(0, 0, (1, num_of_node))
        self.optimizers["hidden_layer"] = AdamOptim()

        # output layer weight
        self.output_layer_w = np.random.uniform(-0.1, 0.1, (num_of_node, self.output_size))
        self.output_layer_b = np.random.uniform(0, 0, (1, 1))
        self.optimizers["output_layer"] = AdamOptim()

    
    def forward(self, x, y_label):
     
        ## x  ---[xW + b]---> y ---[sigmoid]---> z
   
        # print(x.shape)
        self.x = x

        # input layer
        self.y1 = forward_product(self.x, self.input_layer_w, self.input_layer_b)
        self.z1 = forward_sigmoid(self.y1)
        #self.z1 = forward_ReLU(self.y1)

        # hidden layer
        self.y2 = forward_product(self.z1, self.hidden_layer_w, self.hidden_layer_b)
        self.z2 = forward_sigmoid(self.y2)
        #self.z2 = forward_ReLU(self.y2)

        # output layer
        self.y3 = forward_product(self.z2, self.output_layer_w, self.output_layer_b)
        

        # return values
        # self.z3 = forward_sigmoid(self.y3)
        self.z3 = forward_sigmoid(self.y3)

        # loss = y_label - self.z3 # sums up all L2 loss
        loss = np.sum(np.square(y_label - self.z3)) # sums up all L2 loss

        # print(y_label, self.z3)
        
        return loss, self.z3

    def backward(self, y_pred, y_label, lr = 0.1, t = 1, use_opt = True):
        
        # caclulate loss backward
        dL = 1.0*(np.array(y_label).reshape(1, 1) - np.array(y_pred).reshape(1, 1)) # derivative of L2 loss

        # gradient of w3
       
        dLdz2, dLdw3, dLdb3 = backward_product( dLdy = backward_sigmoid(self.y3) * dL, 
                                                   x = self.z2, 
                                                   w = self.output_layer_w )
        # update w3
        if use_opt:
            w_3, b_3 = self.optimizers["output_layer"].update(t, w=self.output_layer_w, b=self.output_layer_b, dw=dLdw3, db=dLdb3)
            self.output_layer_w = w_3
            #self.output_layer_b = b_3
        else:
            self.output_layer_w = dLdw3 * lr + self.output_layer_w
            #self.output_layer_b = dLdb3 * lr + self.output_layer_b

        # gradient of w2
        print("y2",backward_sigmoid(self.y2).shape)
        print("dl",dLdz2.shape)
        dLdz1, dLdw2, dLdb2 = backward_product( dLdy = backward_sigmoid(self.y2)*dLdz2, 
                                                   x = self.z1, 
                                                   w = self.hidden_layer_w )
        # update w2
        if use_opt:
            w_2, b_2 = self.optimizers["hidden_layer"].update(t, w=self.hidden_layer_w, b=self.hidden_layer_b, dw=dLdw2, db=dLdb2)
            self.hidden_layer_w = w_2
            #self.hidden_layer_b = b_2
        else:
            self.hidden_layer_w = dLdw2 * lr + self.hidden_layer_w
            #self.hidden_layer_b = dLdb2 * lr + self.hidden_layer_b

        # gradient of w1
        dLdw1 = np.matmul(self.x.transpose(), dLdz1* backward_sigmoid(self.y1))
        #dLdb1 = dLdz1* backward_ReLU(self.y1)

        # update w1
        self.input_layer_w = self.input_layer_w + lr * dLdw1
        #self.input_layer_b = np.average( self.input_layer_b + lr * dLdb1, axis = 0)


        

# generate data
X_data,Y_data = generate_linear(n=100)
#X_data,Y_data = generate_XOR_easy()

# nn init
nn = simple_NN(input_size = 2, output_size = 1)
nn.build_NN(num_of_node = 250)

# training settings
train_epoch = 2000
lr = 0.1
loss_history = []

# training
for i in range(train_epoch):
    avg_loss = 0
    t = 1
    for x, y in zip(X_data, Y_data):
        
        loss, y_pred = nn.forward(np.array(x).reshape(1, 2), y)
        nn.backward( y_pred, y, lr, t, use_opt = False)

        avg_loss += loss
        t += 1
      
    print(f"Epoch: {i:4d}, loss {avg_loss/X_data.size:.10f}")
    loss_history.append(avg_loss/X_data.size)

print()

# evaluate
y_out = []
correct_count = 0

for x, y in zip(X_data, Y_data):
       
    _, y_pred = nn.forward(np.array(x).reshape(1, 2), y)
    print(f"[y_pred, y_label]: [{y_pred}, {y}]")

    y_out.append(y_pred > 0.5)
    if y_out[-1] == y:
        correct_count += 1
    
print(f"\ncorrect : {correct_count}/{Y_data.size}, acc = {(correct_count/Y_data.size):.2f}")
show_result(X_data, Y_data, y_out)
plt.plot(loss_history)
plt.show()