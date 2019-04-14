#Deep neural network class
import numpy as np
import math
from random import shuffle

class DeepNN:
    L = 0
    units = []
    weights = []
    bias = []
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
            
            self.bias.append(np.zeros(shape = (1,self.units[i]),dtype = float))

#------------------------------ACtivation functions and derivatives------------------------------------
    def sigmoid(self,x):
        #return 1/(1+np.exp(-x))
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

#------------------------------------Metrics---------------------------------------------
    def accuracy(self,y_true,y_pred):
        correct = (y_true==y_pred).sum()

        return correct/y_pred.shape[0]
#----------------------------Loss functions-----------------------------------------------
    
    def BCEcost(self,y_true,y_pred):
        total_loss = np.multiply(y_true,np.log(y_pred)) + np.multiply(1-y_true,np.log(1-y_pred))
        cost = ((-1/y_true.shape[0]) * np.sum(total_loss))
        return cost
#-------------------------------------------------------------------------------------
    def forward_prop(self,x_train):
        caches = [x_train]
        caches_x = [x_train]
        x = x_train
        for i in range(1,len(self.units)):
            z = np.matmul(x,self.weights[i-1].T + self.bias[i-1])
            caches.append(z)
            if(self.activations[i-1]=='relu'):
                x = self.relu(z)
                caches_x.append(x)
            if(self.activations[i-1]=='sigmoid'):
                x = self.sigmoid(z)
                caches_x.append(x)


        return x,caches,caches_x     
    


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


            
    #prediction function
    def predict(self,x_test):
        y_pred_prob = self.forward_prop(x_test)[0]
        y_pred = np.copy(y_pred_prob)
        y_pred[y_pred>=0.5] = 1.0
        y_pred[y_pred<0.5] = 0.0
        return y_pred
        
        
#Training the neural network
       
        
    def train_NN(self,x_train,y_train,epochs,learning_rate,algo_dict,mini_batch_size = 'default'):
       
        
        #Mini batch Gradient Descent, by changing value of batch size, Batch or SGD may be obtained
        def mini_batch_GD(regularization):
            costs = []
            gradient_clipping_value = 6
            
            for i in range(1,epochs+1):
                mini_batches = self.data_loader(x_train,y_train,mini_batch_size)
                for b in mini_batches:
                    x_batch,y_batch = b[0],b[1]
                
                    #forward propogation
                    a,caches,caches_x = self.forward_prop(x_batch)
                    curr_cost = self.BCEcost(y_batch,a)
                    reg_cost = 0
                    for k in range(len(self.weights)):
                        reg_cost = reg_cost + np.linalg.norm(self.weights[k])
                    
                
                    if(i%100==0):
                        costs.append(curr_cost + (regularization*reg_cost)/(2*x_batch.shape[0]))
                        print("Epoch " + str(i))
                        print("Training Cost: " + str(curr_cost) + "\n" + "---------------------" )
                        
                        
                    #backpropogation
                    dA = ((1-y_batch)/(1-a)) - (y_batch/a)
                    for j in range(len(self.weights)-1,-1,-1):
                        dZ = np.multiply(dA,self.derivative(self.activations[j],caches[j+1]))
                        dW = np.matmul(dZ.T,caches_x[j])/x_train.shape[0] + (self.weights[j] * regularization)/(x_batch.shape[0])
                        dB = np.sum(dZ,axis = 0,keepdims=True)/x_batch.shape[0]
                    
                        #Handling gradient explosion
                        if(np.linalg.norm(dW)>=gradient_clipping_value):
                            dW =  (gradient_clipping_value * dW)/np.linalg.norm(dW)
                        
                        dA = np.matmul(dZ,self.weights[j])
                        self.weights[j] = self.weights[j]  - ((learning_rate)*dW)
                        self.bias[j] = self.bias[j] - ((learning_rate)*dB)
                    
            return costs 
        

        #GD with momentum
        def GDMomentum(beta):
            #initializing parameters for RMSprop
            VdW = []
            VdB = []
            for i in range(len(self.weights)):
                vw = np.zeros(shape = self.weights[i].shape,dtype = float)
                vb = np.zeros(shape = self.bias[i].shape,dtype = float)
                VdW.append(vw)
                VdB.append(vb)
            costs = []
            #gradient_clipping_value = 6
            
            for i in range(1,epochs+1):
                mini_batches = self.data_loader(x_train,y_train,mini_batch_size)
                for b in mini_batches:
                    x_batch,y_batch = b[0],b[1]
                
                    #forward propogation
                    a,caches,caches_x = self.forward_prop(x_batch)
                    curr_cost = self.BCEcost(y_batch,a)
                    
                
                    if(i%100==0):
                        costs.append(curr_cost)
                        print("Epoch " + str(i))
                        print("Training Cost: " + str(curr_cost) + "\n" + "---------------------")
                        
                    #backpropogation
                    dA = ((1-y_batch)/(1-a)) - (y_batch/a)
                    for j in range(len(self.weights)-1,-1,-1):
                        dZ = np.multiply(dA,self.derivative(self.activations[j],caches[j+1]))
                        dW = np.matmul(dZ.T,caches_x[j])/x_train.shape[0]
                        dB = np.sum(dZ,axis = 0,keepdims=True)/x_batch.shape[0]
                        
                        VdW[j] = beta*VdW[j] + (1-beta)*dW
                        VdB[j] = beta*VdB[j] + (1-beta)*dB

                        #Handling gradient explosion
                        #if(np.linalg.norm(dW)>=gradient_clipping_value):
                        #    dW =  (gradient_clipping_value * dW)/np.linalg.norm(dW)
                        
                        dA = np.matmul(dZ,self.weights[j])
                        self.weights[j] = self.weights[j]  - ((learning_rate)*VdW[j])
                        self.bias[j] = self.bias[j] - ((learning_rate)*VdB[j])
                    
            return costs     

        #RMSprop algorithm
        def RMSprop(eta):
            #initializing parameters for Momentum update
            SdW = []
            SdB = []
            for i in range(len(self.weights)):
                sw = np.zeros(shape = self.weights[i].shape,dtype = float)
                sb = np.zeros(shape = self.bias[i].shape,dtype = float)
                SdW.append(sw)
                SdB.append(sb)
            costs = []
            #gradient_clipping_value = 6
            
            for i in range(1,epochs+1):
                mini_batches = self.data_loader(x_train,y_train,mini_batch_size)
                for b in mini_batches:
                    x_batch,y_batch = b[0],b[1]
                
                    #forward propogation
                    a,caches,caches_x = self.forward_prop(x_batch)
                    curr_cost = self.BCEcost(y_batch,a)
                    
                
                    if(i%100==0):
                        costs.append(curr_cost)
                        print("Epoch " + str(i))
                        print("Training Cost: " + str(curr_cost) + "\n" +  "----------------" )
                        
                    #backpropogation
                    dA = ((1-y_batch)/(1-a)) - (y_batch/a)
                    for j in range(len(self.weights)-1,-1,-1):
                        dZ = np.multiply(dA,self.derivative(self.activations[j],caches[j+1]))
                        dW = np.matmul(dZ.T,caches_x[j])/x_train.shape[0]
                        dB = np.sum(dZ,axis = 0,keepdims=True)/x_batch.shape[0]
                        
                        SdW[j] = eta*SdW[j] + (1-eta)*(dW**2)
                        SdB[j] = eta*SdB[j] + (1-eta)*(dB**2)

                        #Handling gradient explosion
                        #if(np.linalg.norm(dW)>=gradient_clipping_value):
                        #    dW =  (gradient_clipping_value * dW)/np.linalg.norm(dW)
                        
                        dA = np.matmul(dZ,self.weights[j])
                        self.weights[j] = self.weights[j]  - ((learning_rate*dW)/(SdW[j] + 1e-8)**0.5)
                        self.bias[j] = self.bias[j] - ((learning_rate*dB)/(SdB[j] + 1e-8)**0.5)
                    
            return costs
        
        def Adam(beta1,beta2):
            #Initialize parameters for Adam
            VdW = []
            VdB = []
            SdW = []
            SdB = []
            for i in range(len(self.weights)):
                sw = np.zeros(shape = self.weights[i].shape,dtype = float)
                vw = np.zeros(shape = self.weights[i].shape,dtype = float)
                sb = np.zeros(shape = self.bias[i].shape,dtype = float)
                vb = np.zeros(shape = self.bias[i].shape,dtype = float)
                VdW.append(vw)
                VdB.append(vb)
                SdW.append(sw)
                SdB.append(sb)
            costs = []
            #gradient_clipping_value = 6

            for i in range(1,epochs+1):
                mini_batches = self.data_loader(x_train,y_train,mini_batch_size)
                for b in mini_batches:
                    x_batch,y_batch = b[0],b[1]
                
                    #forward propogation
                    a,caches,caches_x = self.forward_prop(x_batch)
                    curr_cost = self.BCEcost(y_batch,a)
                    
                
                    if(i%100==0):
                        costs.append(curr_cost)
                        print("Epoch " + str(i))
                        print("Training Cost: " + str(curr_cost))
                        print("------------------------------------------")
                        
                        
                    #backpropogation
                    dA = ((1-y_batch)/(1-a)) - (y_batch/a)
                    for j in range(len(self.weights)-1,-1,-1):
                        dZ = np.multiply(dA,self.derivative(self.activations[j],caches[j+1]))
                        dW = np.matmul(dZ.T,caches_x[j])/x_train.shape[0]
                        dB = np.sum(dZ,axis = 0,keepdims=True)/x_batch.shape[0]
                        
                        VdW[j] = beta1*VdW[j] + (1-beta1)*dW
                        VdB[j] = beta1*VdB[j] + (1-beta1)*dB
                        SdW[j] = beta2*SdW[j] + (1-beta2)*(dW**2)
                        SdB[j] = beta2*SdB[j] + (1-beta2)*(dB**2)

                        #Handling gradient explosion
                        #if(np.linalg.norm(dW)>=gradient_clipping_value):
                        #    dW =  (gradient_clipping_value * dW)/np.linalg.norm(dW)
                        
                        dA = np.matmul(dZ,self.weights[j])
                        bias_corrected_vdw = VdW[j]/(1-beta1**i)
                        bias_corrected_vdb = VdB[j]/(1-beta1**i)
                        bias_corrected_sdw = SdW[j]/(1-beta2**i)
                        bias_corrected_sdb = SdB[j]/(1-beta2**i)

                        self.weights[j] = self.weights[j]  - ((learning_rate*bias_corrected_vdw)/(bias_corrected_sdw + 1e-8)**0.5)
                        self.bias[j] = self.bias[j] - ((learning_rate*bias_corrected_vdb)/(bias_corrected_sdb + 1e-8)**0.5)
                    
            return costs 

        

        algo = algo_dict['Algorithm']
        parameters = algo_dict['Parameters']
        if(algo=='mini_batch_GD'):
            training_costs = mini_batch_GD(*parameters)
        if(algo=='GDMomentum'):
            training_costs = GDMomentum(*parameters)
        if(algo=='RMSprop'):
            training_costs = RMSprop(*parameters)
        if(algo=='Adam'):
            training_costs = Adam(*parameters)            
        
        result_dict = {'Trained weights':self.weights,'Trained biases':self.bias,'Training cost':training_costs}

        return result_dict

























        

            

        
          












