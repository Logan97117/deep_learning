#Softmax classification implemented using numpy only
import numpy as np
import math
from random import shuffle


class DeepNN:
    L = 0
    units = []
    weights = []
    biases = []
    activations = []
    
    
    #including the input layer
    def __init__(self,num_layers,units,activation_fns,initialisation):
        self.L = num_layers
        self.units = units
        self.activations = activation_fns

        for i in range(1,len(self.units)):
            if(initialisation=='random'):
                self.weights.append(np.random.randn(self.units[i],self.units[i-1])*0.1)
            if(initialisation=='He'):
                self.weights.append(np.random.randn(self.units[i],self.units[i-1])*math.sqrt(2/self.units[i-1]))
            
            self.biases.append(np.zeros(shape = (1,self.units[i]),dtype = float))


    #----------------Activation functions---------------------------------
    def relu(self,x):
        return np.maximum(x,0.0) 

    def softmax(self,x):
        probs = np.copy(x)
        probs = np.exp(probs-np.max(probs,axis = 1).reshape(x.shape[0],1))
        probs = probs/np.sum(probs,axis = 1).reshape(x.shape[0],1)
        return probs


    def derivative(self,function,x):
        if(function=='relu'):
            a = np.copy(x)
            a[a>=0] = 1.0
            a[a<0] = 0.0
            return a
        if(function=='softmax'):
            grad = np.zeros(shape = (x.shape[0],x.shape[1],x.shape[1]))
            for i in range(x.shape[0]):
                grad[i] = np.multiply(self.softmax(x[i].reshape(1,x.shape[1])),np.identity(x.shape[1]) - self.softmax(x[i].reshape(1,x.shape[1])))

            return grad    
    
    #--------------------Cost function-----------------------------
    def CrossEntropy(self,y_true,y_pred_probs):
        total_loss = np.sum(y_true * np.log(y_pred_probs))
        return (-1/y_true.shape[0]) * total_loss


    #--------------------------------------------------------------
    def forward_prop(self,x_train):
        caches = [x_train]
        caches_x = [x_train]
        x = np.copy(x_train)
        for i in range(1,len(self.units)):
            z = np.matmul(x,self.weights[i-1].T) + self.biases[i-1]
            caches.append(z)
            if(self.activations[i-1]=='relu'):
                x = self.relu(z)
                caches_x.append(x)
            if(self.activations[i-1]=='softmax'):
                x = self.softmax(z)
                caches_x.append(x)
   
        return x,caches,caches_x

    #splitting data into mini batches 
    def data_loader(self,x_train,y_train,mini_batch_size):
        batches = []
        if(mini_batch_size=='default'):
            batches.append((x_train,y_train))
            return batches
        else:
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_tr,y_tr = np.copy(x_train),np.copy(y_train)
            x_tr = x_tr[indices]
            y_tr = y_tr[indices]
            num_batches = math.floor(x_train.shape[0]/mini_batch_size)
            for i in range(num_batches):
                batches.append((x_tr[i*mini_batch_size:(i+1)*mini_batch_size,:],y_tr[i*mini_batch_size:(i+1)*mini_batch_size,:]))
            if(x_train.shape[0]%mini_batch_size!=0):
                batches.append((x_tr[num_batches*mini_batch_size:,:],y_tr[num_batches*mini_batch_size:,:]))
 
            return batches


    def predict(self,x_test):
        x = self.forward_prop(x_test)[0]
        
        y_pred = np.argmax(x,axis = 1)

        return y_pred

    #Mini batch gradient descent implemented only
    def train_NN(self,x_train,y_train,epochs,learning_rate,algo_dict,mini_batch_size = 'default'):
        costs = []
        gradient_clipping_value = 6
        for i in range(1,epochs+1):
            mini_batches = self.data_loader(x_train,y_train,mini_batch_size)
            for b in mini_batches:
                x_batch,y_batch = b[0],b[1]

                #forward propogation
                a,caches,caches_x = self.forward_prop(x_batch)
                curr_cost = self.CrossEntropy(y_batch,a)
                costs.append(curr_cost)

                if(i%100==0):
                    
                    print("Epoch " + str(i))
                    print("Training Cost: " + str(curr_cost) + "\n" + "---------------------" )



                #backpropogation
                dA = np.divide(y_batch,a)
                
                for j in range(len(self.weights)-1,-1,-1):
                    if(self.activations[j]=='softmax'):
                        grad = self.derivative('softmax',caches[j+1])
                        dZ = np.matmul(dA.reshape(x_batch.shape[0],1,dA.shape[1]),grad).reshape(x_batch.shape[0],dA.shape[1])
                    else:    
                        dZ = np.multiply(dA,self.derivative(self.activations[j],caches[j+1]))
                    dW = -1*np.matmul(dZ.T,caches_x[j])/x_batch.shape[0]
                    dB = -1*np.sum(dZ,axis = 0,keepdims=True)/x_batch.shape[0]
                    
                    #Handling gradient explosion
                    if(np.linalg.norm(dW)>=gradient_clipping_value):
                        dW =  (gradient_clipping_value * dW)/np.linalg.norm(dW)

                    dA = np.matmul(dZ,self.weights[j])

                    self.weights[j] = self.weights[j]  - ((learning_rate)*dW)
                    self.biases[j] = self.biases[j] - ((learning_rate)*dB)

        return costs    








