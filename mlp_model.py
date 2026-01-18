import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x)

def derivative_relu(x):
    return np.array(x>0,dtype=np.float32)

def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    x=np.clip(x, -500, 500) # optional improvement to stop overflowing
    return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
    s=sigmoid(x)
    return s*(1-s)

def linear(x):
    return x

def derivative_linear(x):
    return np.ones_like(x)

def softmax(x):
    x_shifted = x - np.max(x, axis=0, keepdims=True)
    expX = np.exp(x_shifted)
    return expX / np.sum(expX, axis=0, keepdims=True)

activation={"relu":(relu,derivative_relu),
            "tanh":(tanh,derivative_tanh),
            "sigmoid":(sigmoid,derivative_sigmoid),
            "linear":(linear,derivative_linear)
            }
#ayers= contains no of neurons in every layers including the input and output
#confg= a dictionary containing what would be the neuron activation and output activation function
def initialize_parameters(layers,config):
    parameters={}
    for l in range(1,len(layers)):
        if(config["activation_function"]=="relu"):
            parameters[f"w{l}"]=np.random.randn(layers[l],layers[l-1])*np.sqrt(2 / layers[l-1])
        elif(config["activation_function"]=="sigmoid" or config["activation_function"]=="tanh"):
            parameters[f"w{l}"]=np.random.randn(layers[l],layers[l-1])* np.sqrt(1 / layers[l-1])
        parameters[f"b{l}"]=np.zeros((layers[l],1))
    
    return parameters

def forward_propagation(x,parameters,config):
    cache={}
    A=x
    L=len(parameters)//2  # here we took half becasue in paramters we have wl and bl for same l we have 2 different paramters hence l=len//2

    for l in range(1,L):
        z=np.dot(parameters[f"w{l}"],A)+parameters[f"b{l}"]
        A=activation[config["activation_function"]][0](z)
        
        cache[f"z{l}"]=z
        cache[f"a{l}"]=A

    ZL=np.dot(parameters[f"w{L}"],A)+parameters[f"b{L}"]

    if(config["output_activation_function"]=="softmax"):
        AL=softmax(ZL)
    else:
        AL=activation[config["output_activation_function"]][0](ZL)
    
    cache[f"z{L}"]=ZL
    cache[f"a{L}"]=AL

    return AL,cache

def cost_fucntion(AL,y,config):
    m=y.shape[1]
    if(config["loss"]=="mse"):
        cost=np.mean((AL-y)**2)
    else:
        cost=-(1/m)*np.sum(y*np.log(AL+1e-9))  #the 1e-9 is added to avoid log(0) case
    
    return cost

def back_propagation(x,y,parameters,cache,config):
    gradients={}
    m=x.shape[1]
    L=len(parameters)//2

    if(config["output_activation_function"]=="softmax"):
        dZ=cache[f"a{L}"]-y
    else:
        dA=cache[f"a{L}"]-y
        dZ=dA*activation[config["output_activation_function"]][1](cache[f"z{L}"])
    
    for l in reversed(range(1,L+1)):
        A_prev=x if l==1 else cache[f"a{l-1}"]
        gradients[f"dW{l}"]=(1/m)*np.dot(dZ,A_prev.T)
        gradients[f"db{l}"]=(1/m)*np.sum(dZ,axis=1,keepdims=True)

        if l>1:
            dA_prev=np.dot(parameters[f"w{l}"].T,dZ)
            dZ=dA_prev * activation[config["activation_function"]][1](cache[f"z{l-1}"])

    return gradients

def update_parameters(parameters,gradients,learning_rate):
    L=len(parameters)//2

    for l in range(1,L+1):
        parameters[f"w{l}"]=parameters[f"w{l}"]-learning_rate*gradients[f"dW{l}"]
        parameters[f"b{l}"]=parameters[f"b{l}"]-learning_rate*gradients[f"db{l}"]

    return parameters

def get_mini_batches(X,y,batch_size=32):
    n_samples=X.shape[0]
    indices=np.arange(n_samples)
    np.random.shuffle(indices)

    for i in range(0,n_samples,batch_size):
        batch_idx = indices[i : i + batch_size]
        
        X_batch = X[:, batch_idx]
        Y_batch = y[:, batch_idx]
        
        yield X_batch, Y_batch

def model(x,y,layers,config,learning_rate,iterations,batch_size=512):
    parameters=initialize_parameters(layers,config)
    cost_list=[]
    for i in range(iterations+1):
        epoch_cost=0
        num_batches=0
        for x_batch,y_batch in get_mini_batches(x,y,batch_size):
            AL,forward_cache=forward_propagation(x_batch,parameters,config)
            cost=cost_fucntion(AL,y_batch,config)
            epoch_cost+=cost
            num_batches+=1
            gradients=back_propagation(x_batch,y_batch,parameters,forward_cache,config)
            parameters=update_parameters(parameters,gradients,learning_rate)
        avg_epoch_cost=epoch_cost/num_batches
        cost_list.append(avg_epoch_cost)

        if(i%(iterations/10)==0):
            print(f"Epoch {i}/{iterations} - Cost: {avg_epoch_cost:.5f}")
    return parameters,cost_list

def accuracy(inp,labels,parameters,config):
    L=len(parameters)//2
    AL,_=forward_propagation(inp,parameters,config)
    a_out=np.argmax(AL,0)
    y_out=np.argmax(labels,0)
    acc=np.mean(a_out==y_out)*100
    return acc