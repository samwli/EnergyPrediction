import numpy as np
from sklearn.linear_model import LinearRegression

def lv1out(x0,y0):
  reg=LinearRegression()
  ys=np.zeros(len(y0))
  for i in range(len(y0)):
    x1=x0[i,:]          #testing data point
    x=np.delete(x0,i,axis=0) #construct the training dataset
    y=np.delete(y0,i,axis=0)
    reg.fit(x,y)        #train the model with x and y

    #reg.predict expects 2D array. array.reshape(1,-1) is used to reshape x1
    y1=reg.predict(x1.reshape(1, -1))  #predict y for the testing point
    ys[i]=y1[:]         #ys is the predicted time series of the predictand
  return ys
