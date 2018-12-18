
# coding: utf-8

# In[1]:


import pandas as pd,numpy as np
from sklearn.preprocessing import minmax_scale,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_validate
import random
from sklearn.metrics import  roc_auc_score
from itertools import  combinations


# In[2]:


def get_data(filename=''):#数据读入
    data=pd.read_csv(r'H:/roadforking/Employee/' + filename,delimiter=',',encoding='utf-8')      
    return data


# In[3]:


def data_Standard(data,Z_score= True): #数据标准化处理
    if Z_score:
        sta=StandardScaler()
        data=sta.fit_transform(data)
    else:
        data=minmax_scale(data)
    return data


# In[4]:


def Dim_asc_mult(data, degree=3):#利用组合的方式对数据进行维度的提升
    new_data = []
    m, n = data.shape
    for indices in combinations(range(n), degree):
        np.set_printoptions(suppress=True)
        new_data.append(np.product(data[:,indices],axis=1))
    return np.array(new_data).T


# In[5]:


def model_auc(x,y,model,cv=5):#利用检验模型,计算auc值
    mean_auc=0
    for i in range(cv): 
        x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=i*random.randint(1,100))
        model.fit(x_tr,y_tr)
        preds = model.predict_proba(x_te)[:, 1]
        auc = roc_auc_score(y_te, preds)
        #print ("AUC (fold %d/%d): %f" % (i + 1, N, auc))
        mean_auc += auc
    return mean_auc/cv


# In[6]:


def data_integ(train_data,test_data):#将测试集与训练集结合起来,便于之后的测试集的预测
    train=np.array(get_data(train_data))
    test=np.array(get_data(test_data))
    M=train.shape[0]
    x_train=train[:,1:]
    y_train=train[:,0]
    x_test=test[:,1:]
    x_all=np.vstack((x_train,x_test))
    x_all=np.delete(x_all,-1,0)
    return x_all,y_train,M


# In[7]:


def fea_data(train_data,test_data,model,sta=False,degree=3,one_hot=False, Dim_asc=False):
    seed=123
    x,y,M=data_integ(train_data,test_data)
    if sta:
        x=data_Standard(x)
    x_de= Dim_asc_mult(x,degree=2)
    x_dt= Dim_asc_mult(x,degree=3)
    x_dim_all=np.hstack((x,x_de,x_dt))
    x_tr_dim=x_dim_all[:M]
    y_tr_dim=y[:M]
    score_set=[]
    need_features = []
    x_try=[]
    i=1
    while  len(score_set)<2 or score_set[-1][0] > score_set[-2][0] :
        scores = []
        for f in range(x_dim_all.shape[1]):
            if f not in need_features:
                fea = list(need_features) + [f]
                if one_hot:
                    x_try=np.hstack(pd.get_dummies(x_tr_dim[:,i]) for i in fea)
                else:
                    x_try=np.vstack([x_tr_dim[:,i] for i in fea])
                    x_try=x_try.T
                score = model_auc(x_try, y, model, N)
                scores.append((f,score))
        print('第%d轮'%i)
        i+=1
        need_features.append(sorted(scores)[-1][0])
        print(need_features)
        score_set.append(sorted(scores)[-1][1])
    need_features.remove(score_set[-1][1])
    need_features = sorted(list(need_features))
    fea_ext=np.hstack([x_dim_all[:,i] for i in need_features]).T
    np.savetxt('fea_ext.txt',fea_ext,delimiter=',',encoding='utf-8')
    return fea_ext,need_features

