#!/usr/bin/env python
# coding: utf-8

# In[5]:


# https://www.kaggle.com/nroman/lgb-single-model-lb-0-9419 ideas referenced from nroman's kaggle kernel
# https://www.kaggle.com/nroman/recursive-feature-elimination
import tensorflow as tf
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, cross_val_score, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import multiprocessing


# In[10]:


files = ['Inputs/train_transaction.csv', 'Inputs/train_identity.csv', 'Inputs/test_transaction.csv', 'Inputs/test_identity.csv', 'Inputs/sample_submission.csv']
def read(input):
    ''' reads all the csv files by TransactionID '''
    data = pd.read_csv(input, index_col='TransactionID')
    return data

with multiprocessing.Pool() as pool:
    train_transaction, train_identity, test_transaction, test_identity, sub = pool.map(read, files)


# In[3]:


def merge(file1, file2):
    ''' merges transaction and identity file '''
    merged = pd.merge(file1, file2, on= 'TransactionID', how='left')#, left_index=True, right_index=True)
    return merged

train = merge(train_transaction, train_identity)
test = merge(test_transaction, test_identity)


# In[4]:


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


'''
Remove useless features
1. Feature has only one value
2. Feature has 85% missing value
3. Feature has 85% or more of same value
4. Correlation between features is not useful???
'''
features = []
feature_list = train.columns
for feature in feature_list:
    if feature == 'isFraud':
        continue
    else:
        if train[feature].nunique() <= 1 or test[feature].nunique() <= 1:
            features.append(feature)
        if (np.count_nonzero(train[feature].isnull()) / len(train[feature]) >= .85) or  (np.count_nonzero(test[feature].isnull()) / len(test[feature]) >= .85):
            features.append(feature)
        if train[feature].value_counts(dropna=False, normalize=True).values[0] >= .85 or test[feature].value_counts(dropna=False, normalize=True).values[0] >= .85:
            features.append(feature)
features = set(features)
#features.remove('isFraud')

print('There will be {} features dropped because they are not useful'.format(len(features)))
train.drop(features, axis=1)
test.drop(features, axis=1)


# In[6]:


for feature in train.columns:
    if train[feature].dtype == 'object':
        encoder = LabelEncoder()
        if feature is 'isFraud':
            encoder.fit(list(train[feature].astype(str).values))
            train[feature] = encoder.transform(list(train[feature].astype(str).values))
        else:
            encoder.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
            train[feature] = encoder.transform(list(train[feature].astype(str).values))
        #test_x[feature] = encoder.transform(list(test_x[feature].values))


# In[7]:


''' resets the index head so that we can drop transactionid after we are sorting '''
train_x = train.sort_values('TransactionDT').reset_index().drop(['TransactionID', 'isFraud', 'TransactionDT'], axis=1)
train_y = train.sort_values('TransactionDT')['isFraud']

train_x.fillna(-99, inplace=True)


# In[8]:


params = {'num_leaves': 491,
          'min_child_weight': 0.02454473273214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 100,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.010283254663721497,
          "boosting_type": "gbdt",
          "bagging_seed": 15,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.2899927210061127,
          'reg_lambda': 0.4485237330340494,
          'random_state': 53
         }


# In[9]:


clf = lgb.LGBMClassifier(**params)
#(n_splits=6, shuffle=False) 'accuracy', 'binary_logloss', 'precision', 'recall'
rfe = RFECV(estimator=clf, step=10, cv=5, scoring='roc_auc', verbose=2)


# In[10]:


rfe.fit(train_x, train_y)


# In[11]:


for col in train_x.columns[rfe.ranking_ == 1]:
    print(col)


# In[14]:


most_influential = pd.DataFrame([col for col in train_x.columns[rfe.ranking_==1]], columns=['features'])
most_influential.to_csv('Import_feature.csv')


# In[8]:


most_influential = pd.read_csv('Inputs/Import_feature.csv')
useful_features = most_influential['features'].tolist()
with multiprocessing.Pool() as pool:
    train_transaction, train_identity, test_transaction, test_identity, sub = pool.map(read, files)

train = merge(train_transaction, train_identity)
test = merge(test_transaction, test_identity)

bad_feature = []
for feature in train.columns:
    if feature not in useful_features:
        bad_feature.append(feature)

bad_feature.remove('isFraud')
bad_feature.remove('TransactionDT')
print(train.shape)
train = train.drop(bad_feature, axis=1)
test = test.drop(bad_feature, axis=1)


# In[117]:


for feature in train.columns:
    if train[feature].dtype == 'object':
        encoder = LabelEncoder()
        if feature is 'isFraud':
            encoder.fit(list(train[feature].astype(str).values))
            train[feature] = encoder.transform(list(train[feature].astype(str).values))
            test[feature] = encoder.transform(list(test[feature].astype(str).values))  
        else:
            encoder.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
            train[feature] = encoder.transform(list(train[feature].astype(str).values))
            test[feature] = encoder.transform(list(test[feature].astype(str).values))  


# In[118]:


''' resets the index head so that we can drop transactionid after we are sorting '''
train_x = train.sort_values('TransactionDT').reset_index().drop(['TransactionID', 'isFraud', 'TransactionDT'], axis=1)
train_y = train.sort_values('TransactionDT')['isFraud']

test = test.sort_values('TransactionDT').reset_index().drop(['TransactionID', 'TransactionDT'], axis=1)


# In[119]:


params = {'num_leaves': 350,
          'min_child_weight': 0.02454473273214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 50,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.010283254663721497,
          "boosting_type": "gbdt",
          "bagging_seed": 15,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha':0.2899927210061127, 
          'reg_lambda': 0.4485237330340494,
          'max_delta_step': 0.5,
          'random_state': 53
         }


# In[120]:


folds = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(folds.split(train_x, train_y)):
    train_data = lgb.Dataset(train_x.iloc[train_idx], label=train_y.iloc[train_idx])
    val_data = lgb.Dataset(train_x.iloc[test_idx], label=train_y.iloc[test_idx])
    model = lgb.train(params, train_data, 10000, valid_sets = [train_data, val_data], verbose_eval=1000, early_stopping_rounds=800)
    


# In[121]:


model.best_iteration


# In[122]:


fraud_model = lgb.LGBMClassifier(**params, num_boost_round=model.best_iteration)
fraud_model.fit(train_x, train_y)


# In[123]:


sub['isFraud'] = fraud_model.predict_proba(test)[:, 1]


# In[124]:


sub.reset_index().to_csv('CSV_Submissions/ieee_cis_fraud_detection_v5.csv', index=False)


# In[ ]:




