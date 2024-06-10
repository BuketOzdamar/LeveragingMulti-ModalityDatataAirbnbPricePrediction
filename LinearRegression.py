# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 17:29:10 2021

@author: BUKET
"""
#kütüphaneler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt #görselleştirme için
import time
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn import preprocessing
#import statsmodels.api as sm  #backward elemination

#veriyukleme
data = pd.read_csv('numeric.csv')
#print(data)

#bağımlı ve bağımsız değişkeni ayırdık.
# kod her çalıştığında veri kümesinin aynı şekilde bölünmesini sağlar.(random_state)
X = data.drop(["fiyati","ilan","sahip","değerlendirme sayısı","son değerlendirme"], axis = 1 ) #ilan adı ve ev sahibi etkilemesin.
y = data["fiyati"]
#print(X)
#eğitim ve test seti ayırma.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=0)

#LinearRegression için model oluşturma
lr_model = LinearRegression()
model = lr_model.fit(X_train,y_train)

y_pred = lr_model.predict(X_test)

#grafiğimizi düzgün alabilmek için index numaralarına göre sıralattık.
X_train = X_train.sort_index()
y_train = y_train.sort_index()
y_test = y_test.sort_index()

from sklearn import metrics
print("------------------")
print("Linear r^2 degeri")
print("------------------")
print(r2_score(y_train,lr_model.predict(X_train)))
print("------------------")
print("Linear Regression RMSE degeri")#Root Mean Square Error(RMSE)
print("------------------------------")
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("------------------------------")

plt.figure(figsize=(10,8))
#plt.scatter(y_test,y_pred,color='black')
plt.plot(y_test,color='red')
plt.plot(y_pred,color='blue')
plt.title("")
plt.show

'''
import statsmodels.api as sm

X_1 = np.append(arr = np.ones((313,1)).astype(int), values=X, axis = 1)

X_2 = X.iloc[:,[0,1,2,3,4,5,6,7]].values #hepsinin pvaluesını hesaplayıp sonuca göre yüksek olanı eleyeceğiz.
X_2 = np.array(X_2, dtype=float)
model = sm.OLS(y,X_2).fit()
print(model.summary())

X = X.iloc[:,[0,1,4,5,6,7]]#2(balkon), 3(yorum) ve 7(temizlik) indekslerini çıkarttık.

X_1 = np.append(arr = np.ones((313,1)).astype(int), values=X, axis = 1)

X_2 = X.iloc[:,[0,1,2,3,4,5]].values
X_2 = np.array(X_2, dtype=float)
model = sm.OLS(y,X_2).fit()
print(model.summary())


X = X.iloc[:,[0,1,4,5,6]]

X_1 = np.append(arr = np.ones((313,1)).astype(int), values=X, axis = 1)

X_2 = X.iloc[:,[0,1,2,3,4]].values #hepsinin pvaluesını hesaplayıp sonuca göre yüksek olanı eleyeceğiz.
X_2 = np.array(X_2, dtype=float)
model = sm.OLS(y,X_2).fit()
print(model.summary())

#backward elemination yönteminden sonra tekrar eğitelim.
yeniX = data.drop(["balkon","yorum","temizlik","fiyati","ilan adi","ev sahibi","değerlendirme","son değerlendirme"], axis=1)
y = data["fiyati"]
yeniX_train, yeniX_test, y_train, y_test = train_test_split(yeniX,y,test_size=0.35, random_state=20)

lr_model = LinearRegression()
model = lr_model.fit(yeniX_train,y_train)

y_pred = lr_model.predict(yeniX_test)

sns.distplot(y)
plt.sho

from sklearn.model_selection import cross_val_score
basarı = cross_val_score(estimator=lr_model, X=yeniX_train, y=y_train, cv=4)
print(basarı.mean())
'''






































