import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import pickle

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    c=np.max(x,axis=-1, keepdims=True)
    x=x-c
    return np.exp(x)/np.sum(np.exp(x),axis=-1, keepdims=True)

def get_data():
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl","rb") as f:
        network=pickle.load(f)
    return network

def predict(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=softmax(a3)
    return y

x, t = get_data()
network = init_network()
accuracy_cnt=0
for i in range(0,len(x),100):
    y=predict(network, x[i:i+100])
    p=np.argmax(y,axis=1)
    accuracy_cnt+=np.sum(p==t[i:i+100])
        
print(len(x))
print("Acc:"+str(float(accuracy_cnt)/len(x)))