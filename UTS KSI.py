#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout


# In[19]:


data = pd.read_csv(r"C:\Users\haikal\Downloads\maret 2023\AAPL.csv")
data.head()


# In[20]:


data.info()


# In[21]:


data["Close"]=pd.to_numeric(data.Close,errors='coerce')
data = data.dropna()
trainData = data.iloc[:,4:5].values


# In[22]:


data.info()


# In[6]:


sc = MinMaxScaler(feature_range=(0,1))
trainData = sc.fit_transform(trainData)
trainData.shape


# In[23]:


X_train = []
y_train = []

for i in range (60,10468):
    X_train.append(trainData[i-60:i,0]) 
    y_train.append(trainData[i,0])

X_train,y_train = np.array(X_train),np.array(y_train)


# In[24]:


X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1)) #adding the batch_size axis
X_train.shape


# In[25]:


model = Sequential()

model.add(LSTM(units=100, return_sequences = True, input_shape =(X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(units =1))
model.compile(optimizer='adam',loss="mean_squared_error")


# In[27]:


hist = model.fit(X_train, y_train, epochs = 20, batch_size = 32, verbose=2)


# In[28]:


plt.plot(hist.history['loss'])
plt.title('Training model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# In[30]:


testData = pd.read_csv(r"C:\Users\haikal\Downloads\maret 2023\AAPL.csv")
testData["Close"]=pd.to_numeric(testData.Close,errors='coerce')
testData = testData.dropna()
testData = testData.iloc[:,4:5]
y_test = testData.iloc[60:,0:].values 
#input array for the model
inputClosing = testData.iloc[:,0:].values 
inputClosing_scaled = sc.transform(inputClosing)
inputClosing_scaled.shape
X_test = []
length = len(testData)
timestep = 60
for i in range(timestep,length):  
    X_test.append(inputClosing_scaled[i-timestep:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
X_test.shape


# In[31]:


y_pred = model.predict(X_test)
y_pred


# In[32]:


predicted_price = sc.inverse_transform(y_pred)


# In[35]:


plt.plot(y_test, color = 'red', label = 'Harga Asli Saham')
plt.plot(predicted_price, color = 'green', label = 'Harga Prediksi Saham')
plt.title('Prediksi Harga Saham Apple')
plt.xlabel('Waktu')
plt.ylabel('Harga Saham')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




