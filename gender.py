import pandas as pd
import numpy as np



data = pd.read_csv("C:/Users/sn/Desktop/heights_weights_genders.csv")
data

def G(Gender):
    if 'Female' in Gender:
        return 0
    else:
        return 1
data['Gender'] = data['Gender'].apply(G)

x = data.drop(columns=['Gender']).to_numpy()
N=x.shape
y = data['Gender'].to_numpy()
x=np.concatenate((x,np.ones((N[0],1))),axis=1)

y=y.reshape(N[0],1)
y=y.astype(float)

def sigmoid(w,x):
  s=1/(1+np.exp(-np.matmul(x,w)))
  return s.reshape(N[0],1)
def computeloss(w,x,y):
  y1=np.transpose(y)
  x1=np.log(sigmoid(w,x))
  y2=np.transpose(1-y)
  x2=np.log(1-sigmoid(w,x))
  l=-(np.matmul(y1,x1)+np.matmul(y2,x2))
  return l.reshape(1)

w_0=np.random.rand(3,1)/500
epoch=100
learning_rate=0.00000002
cost=[]
for i in range(3000):
  
  h=sigmoid(w_0,x)
  a=np.transpose(h-y)
  w_new=w_0-learning_rate*np.matmul(a,x).reshape(3,1)
  c=computeloss(w_0,x,y)
  cost.append(computeloss(w_0,x,y))
  w_0=w_new


y_hat=sigmoid(w_new,x)
y_hat[y_hat>=0.5]=1
y_hat[y_hat<0.5]=0
accuracy=np.sum((1-np.abs(y_hat-y))/N[0])
print(accuracy)

def sgn(x):
  x[x==0]=1
  return x
y[y==0]=-1

w = np.zeros((3,1))
n_miss_list = []
epochs=300
lr=1
for epoch in range(epochs):
  y_hat=sgn(np.sign(np.matmul(x,w)))
  w+=lr*np.sum((y-y_hat)*x,axis=0).reshape(3,1)

accuracy=np.sum(y_hat==y)/N[0]
print(accuracy)