# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 23:23:43 2022

@author: lalith kumar
"""
# Preparing a document for model explaining how many dummy variables have created and RMSE value for  model.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib
from matplotlib.pylab import rcParams
rcParams["figure.figsize"]=15,6
import statsmodels.graphics.tsaplots as tsa_plots

# IMPORTING DATASET.

data=pd.read_excel('E:\\data science\\ASSIGNMENTS\\ASSIGNMENTS\\forecasting\\CocaCola_Sales_Rawdata.xlsx')
data
cc=data.copy()
cc
cc.shape#(42, 2)
cc.describe()
cc.info()
cc.isnull().sum()
cc.head()
cc.columns
cc.describe().T

#TSAplot
tsa_plots.plot_acf(cc.Sales)

import numpy as np
quarter=['Q1','Q2','Q3','Q4']
n=cc['Quarter'][2]
n[0:2]

cc['quarter']=0

for i in range(42):
    n=cc['Quarter'][i]
    cc['quarter'][i]=n[0:2]

#GET DUMMIES
dummy=pd.DataFrame(pd.get_dummies(cc['quarter']))

coco=pd.concat((cc,dummy),axis=1)
t= np.arange(1,43)
coco['t']=t
coco['t_square']=coco['t']*coco['t']

log_Sales=np.log(coco['Sales'])
coco['log_Sales']=log_Sales

train= coco.head(78)
test=coco.tail(5)
coco.Sales.plot()

import statsmodels.formula.api as smf

#linear model
linear= smf.ols('Sales~t',data=train).fit()
predlin=pd.Series(linear.predict(pd.DataFrame(test['t'])))
rmselin=np.sqrt((np.mean(np.array(test['Sales'])-np.array(predlin))**2))
rmselin.round(2)  

#quadratic model
quad=smf.ols('Sales~t+t_square',data=train).fit()
predquad=pd.Series(quad.predict(pd.DataFrame(test[['t','t_square']])))
rmsequad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predquad))**2))
rmsequad.round(2) 

#exponential model
expo=smf.ols('log_Sales~t',data=train).fit()
predexp=pd.Series(expo.predict(pd.DataFrame(test['t'])))
predexp
rmseexpo=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predexp)))**2))
rmseexpo.round(2) 

#additive seasonality
additive= smf.ols('Sales~ Q1+Q2+Q3+Q4',data=train).fit()
predadd=pd.Series(additive.predict(pd.DataFrame(test[['Q1','Q2','Q3','Q4']])))
predadd
rmseadd=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predadd))**2))
rmseadd.round(2) 

#additive seasonality with linear trend
addlinear= smf.ols('Sales~t+Q1+Q2+Q3+Q4',data=train).fit()
predaddlinear=pd.Series(addlinear.predict(pd.DataFrame(test[['t','Q1','Q2','Q3','Q4']])))
predaddlinear

rmseaddlinear=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predaddlinear))**2))
rmseaddlinear.round(2) 

#additive seasonality with quadratic trend
addquad=smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=train).fit()
predaddquad=pd.Series(addquad.predict(pd.DataFrame(test[['t','t_square','Q1','Q2','Q3','Q4']])))
rmseaddquad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predaddquad))**2))
rmseaddquad.round(2) 

#multiplicative seasonality
mulsea=smf.ols('log_Sales~Q1+Q2+Q3+Q4',data=train).fit()
predmul= pd.Series(mulsea.predict(pd.DataFrame(test[['Q1','Q2','Q3','Q4']])))
rmsemul= np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmul)))**2))
rmsemul.round(2) 

#multiplicative seasonality with linear trend
mullin= smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data=train).fit()
predmullin= pd.Series(mullin.predict(pd.DataFrame(test[['t','Q1','Q2','Q3','Q4']])))
rmsemulin=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmullin)))**2))
rmsemulin.round(2) 

#multiplicative seasonality with quadratic trend
mul_quad= smf.ols('log_Sales~t+t_square+Q1+Q2+Q3+Q4',data=train).fit()
pred_mul_quad= pd.Series(mul_quad.predict(test[['t','t_square','Q1','Q2','Q3','Q4']]))
rmse_mul_quad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_mul_quad)))**2))
rmse_mul_quad.round(2) 

#tabulating the rmse values

data={'Model':pd.Series(['rmse_mul_quad','rmseadd','rmseaddlinear','rmseaddquad','rmseexpo','rmselin','rmsemul','rmsemulin','rmsequad']),'Values':pd.Series([rmse_mul_quad,rmseadd,rmseaddlinear,rmseaddquad,rmseexpo,rmselin,rmsemul,rmsemulin,rmsequad])}
data

Rmse=pd.DataFrame(data)
Rmse
Rmse.sort_values(['Values'])

'''
Out[32]: 
           Model       Values
7      rmsemulin   185.330435
3    rmseaddquad   201.355104
0  rmse_mul_quad   225.096259
2  rmseaddlinear   351.878960
5        rmselin   372.201207
8       rmsequad   445.059966
4       rmseexpo   499.467926
1        rmseadd  1665.791566
6        rmsemul  1807.652663
'''

#-----------------------------------------------------------------------------















