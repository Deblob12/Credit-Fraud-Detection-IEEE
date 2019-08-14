#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras 
import pandas as pd
import numpy as np 
import matplotlib as mp
from sklearn import preprocessing 


# In[2]:


def read(input):
    ''' reads all the csv files by TransactionID '''
    data = pd.read_csv(input, index_col='TransactionID')
    return data


# In[3]:


def mergesort(file1, file2):
    ''' merges transaction and identity file '''
    merged = pd.merge(file1, file2, how='left', left_index=True, right_index=True)
    return merged


# In[4]:


def read_merge_train(file1, file2):
    train_transaction = read(file1)
    train_identity = read(file2)
    merged_train = mergesort(train_transaction, train_identity)
    return merged_train

def read_merge_test(file1, file2):    
    test_transaction = read(file1)
    test_identity = read(file2)
    merged_test = mergesort(test_transaction, test_identity)
    return merged_test


# In[ ]:


def ratio(data):
    fraud = 0
    for num in data['isFraud']:
        if num == 1:
            fraud += 1
    non_fraud = data.shape[0]
    _ratio = np.true_divide(non_fraud, fraud)
    _ratio = _ratio
    return {0:1., 1:_ratio}
    


# In[5]:


train = read_merge_train('train_transaction.csv', 'train_identity.csv')
test = read_merge_test('test_transaction.csv', 'test_identity.csv')
train_x = train.drop('isFraud', axis=1)
train_y = train['isFraud'].copy()
# test_x = test.drop('isFraud', axis=1)
test_x = test
# test_y = test['isFraud'].copy()
print(train.shape)
print(test.shape)


# In[35]:


''' split the data by fraud or non fraud for data cleaning '''
df_non_fraud=train.loc[train['isFraud']==0]
df_fraud=train.loc[train['isFraud']==1]


# In[36]:


df_non_fraud


# In[7]:


train_y.shape


# In[8]:


for feature in train_x.columns:
    if train_x[feature].dtype == 'object' or test_x[feature].dtype == 'object':
        encoder = preprocessing.LabelEncoder()
        encoder.fit(list(train_x[feature].values) + list(test_x[feature].values))
        train_x[feature] = encoder.transform(list(train_x[feature].values))
        test_x[feature] = encoder.transform(list(test_x[feature].values))


# In[25]:


train_x


# In[33]:


for x in train_x['id_32']:
    if np.isnan(x):
        print(x)


# In[9]:


# train_x_mod = np.array(train_x.head(n=540540)).astype(float)

# train_y_mod = np.array(train_y[:540540]).astype(np.int32)
train_x_mod = np.array(train_x).astype(float)

train_y_mod = np.array(train_y).astype(np.int32)

# test_x_mod = np.array(train_x.tail(n=50000)).astype(float)

# test_y_mod = np.array(train_y[540540:]).astype(np.int32)


# In[10]:


print(train_x_mod.shape)
print(train_y_mod.shape)


# In[11]:


a = train_x_mod.reshape(590540, 1, 432)
a.shape


# In[30]:


print(len(train_x.isnull().any()))


# In[41]:


weights = ratio(train)
print(weights)
fraud_model = keras.models.Sequential()
fraud_model.add(keras.layers.LSTM(128, input_shape=(None, 432), return_sequences=True))
fraud_model.add(keras.layers.Activation('relu'))
#fraud_model.add(keras.layers.Embedding(input_dim=432, output_dim=64))
fraud_model.add(keras.layers.LSTM(64, input_shape=(None, 432)))#, return_sequences=True))
fraud_model.add(keras.layers.Dense(128, input_shape=(None, 432)))
fraud_model.add(keras.layers.Dropout(.5))
fraud_model.add(keras.layers.Activation('relu'))
fraud_model.add(keras.layers.Dense(2,activation='softmax'))
fraud_model.compile(loss = 'sparse_categorical_crossentropy', optimizer='nadam',metrics = ['accuracy'])#, loss_weights = [.5])
fraud_model_complete = fraud_model.fit(a, train_y_mod, batch_size = 512, epochs = 2, validation_split=.25, class_weight=weights) #validation_data = (x_test, y_test), callbacks = [stop, checkpoint])#, use_multiprocessing= True)
#fraud_model_complete.evaluatae()


# In[ ]:




