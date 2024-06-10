# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 21:02:17 2021

@author: BUKET
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

data = pd.read_csv('numeric.csv')

#bağımlı ve bağımsız değişkeni ayırdık.
X = data.drop(["fiyati","ilan","sahip","değerlendirme sayısı","son değerlendirme"], axis = 1 ) #ilan adı ve ev sahibi etkilemesin.
y = data["fiyati"]

#eğitim ve test seti ayırma.
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=60)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X1 = sc_X.fit_transform(X)
#◘fit fonksiyonu model inşaasını uygulamaya çalışıyor.
sc_y = StandardScaler()
y1 = sc_y.fit_transform(y.values.reshape(-1,1))


# SVR modeli eğitme
svr_reg = SVR(kernel='rbf')#değişkenler arasındaki ilişki durumuna göre bir çekirdek seçilmeli
reg = svr_reg.fit(X1, y1)

y_pred = svr_reg.predict(X1)

from sklearn import metrics
print("------------------")
print("SVR r^2 degeri")
print("------------------")
print(r2_score(y1,y_pred))
print("------------------")
print("SVR RMSE degeri")#Root Mean Square Error(RMSE)
print("------------------")
print(np.sqrt(metrics.mean_squared_error(y1,y_pred)))
print("------------------")

plt.figure(figsize=(30,8))
plt.plot(y_pred, color='red')
#plt.scatter(y1,y_pred,color='r')
plt.plot(y1,color='blue')
plt.show







































