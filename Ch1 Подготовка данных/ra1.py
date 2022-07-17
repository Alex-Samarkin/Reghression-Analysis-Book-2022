## -*- coding: UTF-8 -*-

"""
    Первичный анализ и проверка данных
    Самаркин А.И.
    15/07/2022
    учебный файл
"""

# импорт модулей python
import datetime as dt
from genericpath import exists
import statistics as sts

import numpy as np
import scipy as sp
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels as sm
import sklearn as sklearn


hr = "_"*100
now_is = dt.datetime.now()
print(f"Текущая дата и время: {now_is}")

url = "https://raw.githubusercontent.com/Alex-Samarkin/Reghression-Analysis-Book-2022/main/Ch1%20%D0%9F%D0%BE%D0%B4%D0%B3%D0%BE%D1%82%D0%BE%D0%B2%D0%BA%D0%B0%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85/heart_failure_clinical_records_dataset_orig.csv"
file_name = "local_data.csv"

data : pd.DataFrame

if exists(file_name):
    data = pd.read_csv(file_name)
else:
    data = pd.read_csv(url)
    pass

print(hr)
print(data.head())
print("...")
print(data.tail())
print(hr)
data.to_csv(file_name, index=False)

# в буфер обмена - раскомментируйте следующую строку
# data.to_clipboard()

# первичная статистика
print(data.describe())

data.hist()
plt.show()

for i in data.columns:
    sns.scatterplot(data=data[i])
    # plt.show()

# стандартизация данных
data1 = data.iloc[:,[0,2,4,6,7,8,11]]

sns.boxplot(data=data1)
plt.show()

# стандартизация данных
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
 
df_scaled = std_scaler.fit_transform(data1.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=[data1.columns])
sns.boxplot(data=df_scaled)
plt.show()

df_scaled.hist()
plt.show()

# преобразование Box-Cox
from sklearn.preprocessing import PowerTransformer

data_np = data1.to_numpy()
col_names = data1.columns;
pt = PowerTransformer()
pt.fit(data_np)
print(pt.lambdas_)
# pt.transform
data_np = pt.fit_transform(data_np)
data_np = pd.DataFrame(data_np,columns=col_names)

print(data_np.head())
sns.boxplot(data=data_np)
plt.show()

# удаление выбросов
z1 = np.abs(sp.stats.zscore(data1))
z_np = np.abs(sp.stats.zscore(data_np))

out1 = z1[z1>3]
out_np = z_np[z_np>3]

print()
print(out1.head(10))
print()
print(out_np.head(10))

sns.heatmap(z1)
plt.show()
sns.heatmap(z_np)
plt.show()

sns.heatmap(out1)
plt.show()
sns.heatmap(out_np)
plt.show()

# поиск аномалий методом изолирующего леса и локальной величины выброса
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(data_np)

from sklearn.neighbors import LocalOutlierFactor
lout = LocalOutlierFactor(n_neighbors=20)
yhat2 = lout.fit_predict(data_np)

flag   = pd.DataFrame([yhat,yhat2])
flag = flag.T
print( flag.head() )
sns.scatterplot(data=yhat)
sns.scatterplot(data=yhat2)
plt.show()

data_np["flag1"] = yhat
data_np["flag2"] = yhat2
sns.scatterplot(data=data_np, x="age", y="time", hue="flag1")
plt.show()
sns.scatterplot(data=data_np, x="age", y="time", hue="flag2")
plt.show()

# расчет коэффициентов корреляции и коэффициентов достоверности
data_np.drop(["flag1","flag2"],axis=1,inplace=True)
r = data_np.corr()

print(r)
sns.heatmap(r,annot=True)
plt.show()

# кластерный анализ
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit_transform(data_np)
data_np["CLUSTER"] = kmeans.labels_
print(kmeans.cluster_centers_)
sns.scatterplot(data=data_np, x="age", y="time", hue="CLUSTER")
plt.show()

#PCA
from sklearn.decomposition import PCA
data_np.drop("CLUSTER", axis= 1, inplace= True)

pca = PCA(n_components=4)
pca.fit(data_np)
data_PCA = pca.transform(data_np)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)
plt.plot(pca.explained_variance_ratio_)
plt.show()

#Factor
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=3, rotation="varimax")
fa.fit(data_np)
data_fa = fa.transform(data_np)
print(fa.components_.T)
plt.show()



