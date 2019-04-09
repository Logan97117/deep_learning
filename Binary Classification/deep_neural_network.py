#Deep neural network class
import numpy as np

class DeepNN:
    L = 0
    units = []
    weights = []
    bias = []
    activations = []

    def __init__(self,num_layers,units,activation_fns):
        self.L = num_layers
        self.units = units
        self.activations = activation_fns

        for i in range(1,len(self.units)):
            self.weights.append(np.random.randn(self.units[i],self.units[i-1])*0.1)
            self.bias.append(np.zeros(shape = (1,self.units[i]),dtype = float))

    #Two activation functions are there, more will be added and another module of activations will be there
    
    def sigmoid(self,x):
        return np.where(x >= 0,1 / (1 + np.exp(-x)),np.exp(x) / (1 + np.exp(x)))
        

    def relu(self,x):
        return np.maximum(x,0)

    def derivative(self,function,x):
        if(function=='sigmoid'):
            return self.sigmoid(x) * (1-self.sigmoid(x))
        if(function=='relu'):
            x[x>=0] = 1.0
            x[x<0] = 0.0
            return x


    def accuracy(self,y_true,y_pred_prob):
        y_pred = np.copy(y_pred_prob)
        y_pred[y_pred>=0.5] = 1
        y_pred[y_pred<0.5] = 0
        correct = (y_true==y_pred).sum()

        return correct/y_pred.shape[0]


    def forward_prop(self,x_train):
        caches = [x_train]
        caches_x = [x_train]
        x = x_train
        for i in range(1,len(self.units)):
            z = np.dot(x,self.weights[i-1].T + self.bias[i-1])
            caches.append(z)
            if(self.activations[i-1]=='relu'):
                x = self.relu(z)
                caches_x.append(x)
            if(self.activations[i-1]=='sigmoid'):
                x = self.sigmoid(z)
                caches_x.append(x)


        return x,caches,caches_x     

    #Binary Cross Entropy loss
    def cost(self,y_true,y_pred):
        total_loss = np.multiply(y_true,np.log(y_pred)) + np.multiply(1-y_true,np.log(1-y_pred))
        cost = (-1/y_true.shape[0]) * np.sum(total_loss)
        return cost

    #Training the Neural network
    def train_NN(self,x_train,y_train,iters,learning_rate,x_val,y_val):

        costs = []
        for i in range(1,iters+1):
            
            #forward propogation
            a,caches,caches_x = self.forward_prop(x_train)
            curr_cost = self.cost(y_train,a)
            costs.append(curr_cost)
            
            if(i%100==0):
                print("Epoch " + str(i))
                a_y = self.forward_prop(x_val)[0]
                print("Training Cost: " + str(curr_cost) + "------------" + "Validation cost: " + str(self.cost(y_val,a_y)))
                print("Training Accuracy: " + str(self.accuracy(y_train,a)))
                print("Validation Accuracy: " + str(self.accuracy(y_val,a_y)))
                print("-----------------------------------------------------------")
                

            
            #backpropogation
            dA = ((1-y_train)/(1-a)) - (y_train/a)
            for j in range(len(self.weights)-1,-1,-1):
                dZ = np.multiply(dA,self.derivative(self.activations[j],caches[j+1]))
                dW = np.dot(dZ.T,caches_x[j])/x_train.shape[0]
                dB = np.sum(dZ,axis = 0,keepdims=True)/x_train.shape[0]
                '''if(np.linalg.norm(dW)>=4):
                    dW = dW/np.linalg.norm(dW)
                if(np.linalg.norm(dB)>=1.5):
                    dB = dB/np.linalg.norm(dB) 

                '''
                dA = np.dot(dZ,self.weights[j])
                self.weights[j] = self.weights[j] - ((learning_rate)*dW)
                self.bias[j] = self.bias[j] - ((learning_rate)*dB)
                
                
        out_dict = {'Trained weights':self.weights,'Trained biases':self.bias,'Training cost':costs} 

        return out_dict 








        

            

        
          












