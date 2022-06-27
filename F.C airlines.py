# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:55:05 2022

@author: lalith kumar
"""
# Preparing a document for model explaining how many dummy variables  have created and RMSE value for model.
# importing datasets.

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('E:\\data science\\ASSIGNMENTS\\ASSIGNMENTS\\forecasting\\Airlines+Data.xlsx')

df.shape
list(df)
df.info()
df.describe()
df.Passengers.plot()
df

# for 't'

t=[]
for i in range(1,97,1):
    t.append(i)

df['t']=pd.DataFrame(t)
df['t_square']=np.square(df['t'])
df['log_pass']=np.log(df['Passengers'])

#Extracting months
df['month']=df['Month'].dt.strftime('%b')

#Extracting years
df['year']=df['Month'].dt.strftime('%Y')

#Getting dummies of month column
df=pd.get_dummies(df,columns=['month'])
df['month']=df['Month'].dt.strftime('%b')
list(df)

#Plots
#Heatmap
%matplotlib qt
plt.figure(figsize=(12,8))
heatmap_month=pd.pivot_table(data=df,values='Passengers',index='year',columns='month',aggfunc='mean',fill_value=0)
sns.heatmap(heatmap_month,annot=True,fmt='g')

#Boxplot
%matplotlib qt
plt.figure(figsize=(12,8))
sns.boxplot(x='month',y='Passengers',data=df)
sns.boxplot(x='year',y='Passengers',data=df)


#Line plot
%matplotlib qt
plt.figure(figsize=(12,8))
sns.lineplot(x='month',y='Passengers',data=df)
sns.lineplot(x='year',y='Passengers',data=df)

# Splitting data
df.shape
Train = df.head(77)
Test = df.tail(19)

#Fitting the model 
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error


#Linear Model

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
#np.sqrt(np.mean((Test['Passengers']-np.array(pred_linear))**2))
rmse_linear
# rmse_linear
# Out[101]: 58.14854431950883

#Exponential
Exp = smf.ols('Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
# rmse_Exp
# Out[102]: 4.402341105643053e+129

#Quadratic 
Quad = smf.ols('Passengers~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
#pred_Quad = pd.Series(Exp.predict(pd.DataFrame(Test[["t","t_square"]))) # we hve to verify
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad
# rmse_Quad
# Out[103]: 58.926328528189

#Additive seasonality 
add_sea = smf.ols('Passengers~month_Apr+month_Aug+month_Dec+month_Feb+month_Jan+month_Jul+month_Jun+month_Mar+month_May+month_Nov+month_Oct+month_Sep',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['month_Apr','month_Aug','month_Dec','month_Feb','month_Jan','month_Jul','month_Jun','month_Mar','month_May','month_Nov','month_Oct','month_Sep']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea
# rmse_add_sea
# Out[104]: 133.3154036011272

#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Passengers~t+t_square+month_Apr+month_Aug+month_Dec+month_Feb+month_Jan+month_Jul+month_Jun+month_Mar+month_May+month_Nov+month_Oct+month_Sep',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['t','t_square','month_Apr','month_Aug','month_Dec','month_Feb','month_Jan','month_Jul','month_Jun','month_Mar','month_May','month_Nov','month_Oct','month_Sep']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad
# rmse_add_sea_quad
# Out[105]: 39.61752923080081

#Multiplicative Seasonality
Mul_sea = smf.ols('Passengers~month_Apr+month_Aug+month_Dec+month_Feb+month_Jan+month_Jul+month_Jun+month_Mar+month_May+month_Nov+month_Oct+month_Sep',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
# rmse_Mult_sea
# Out[106]: 2.2435447278121173e+95

#Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('log_pass~t+month_Apr+month_Aug+month_Dec+month_Feb+month_Jan+month_Jul+month_Jun+month_Mar+month_May+month_Nov+month_Oct+month_Sep',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 
# rmse_Mult_add_sea
# Out[107]: 12.183266271900296

#Compare the results 
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
type(data)

table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])

'''
 Out[96]: 
               MODEL    RMSE_Values
6  rmse_Mult_add_sea   1.218327e+01
4  rmse_add_sea_quad   3.961753e+01
0        rmse_linear   5.814854e+01
2          rmse_Quad   5.892633e+01
3       rmse_add_sea   1.333154e+02
5      rmse_Mult_sea   2.243545e+95
1           rmse_Exp  4.402341e+129
'''

#-------------------------------------------------------------------------------------------------



