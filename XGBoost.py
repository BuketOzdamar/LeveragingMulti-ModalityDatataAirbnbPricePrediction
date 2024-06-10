# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 23:11:27 2021

@author: BUKET
"""
#kütüphaneler
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt #görselleştirme için
from sklearn import model_selection
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn import metrics
#veriyukleme
data = pd.read_csv('numeric.csv')

X = data.drop(["fiyati","ilan","sahip","değerlendirme sayısı","son değerlendirme"], axis = 1 ) #ilan adı ve ev sahibi etkilemesin.
y = data["fiyati"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=64) #en yüksek test_size ve random_state


xgb_model = XGBClassifier()
#xgb1 = XGBRegressor(colsample_bytree = 0.5, learning_rate = 0.09, max_depth = 4, n_estimators = 2000)
model = xgb_model.fit(X_train,y_train)

y_pred = xgb_model.predict(X_test)


#grafiğimizi düzgün alabilmek için index numaralarına göre sıralattık.
X_train = X_train.sort_index()
y_train = y_train.sort_index()
y_test = y_test.sort_index()

print("------------------")
print("XGBoost r^2 degeri")
print("------------------")
print(r2_score(y_train,xgb_model.predict(X_train)))
print("------------------")
print("XGBoost RMSE degeri")#Root Mean Square Error(RMSE)
print("------------------------------")
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
#print(mean_squared_error(y_test, y_pred, squared=False))
#print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("------------------------------")

plt.figure(figsize=(12,8))
#plt.scatter(y_test,y_pred,color='r')
plt.plot(y_pred,color='blue')
plt.plot(y_test,color='r')

plt.show

'''
sns.distplot(y)
plt.sho
'''
'''


from xgboost import XGBClassifier
xgb_clf = XGBClassifier(n_estimators=20, learning_rate=0.25, max_features=7, max_depth=2, random_state=0, use_label_encoder=False)
xgb_clf.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
print("Extreme Gradient Boosting")
'''

























