## -*- coding: UTF-8 -*-

"""
    _summary_
    _author_
    _date_
    _purpose_
"""

# импорт модулей python
import datetime as dt
import statistics as sts
# импорт научных модулей
import numpy as np
import scipy as sp
import pandas as pd
# импорт графики
import matplotlib.pyplot as plt
import seaborn as sns
# импорт модулей регрессионного анализа
import statsmodels as sm
import sklearn as sk


hr = "-"*100

# тест модулей python
print(hr)
now_is = dt.datetime.now()
print(f"Текущая дата и время: {now_is}")

# тест научных модулей, печатаем или версию или документацию
print(hr)
print("Статистика "+sts.__doc__)
print("Numpy "+np.__version__)
print("Scipy "+sp.__version__)
print("Pandas "+pd.__version__)
print("Matplotlib "+plt.__name__)
print("Seaborn "+sns.__version__)
print("Statsmodels "+sm.__version__)
print("Sklearn  "+sk.__version__)

# проверка векторов и графики
t = np.linspace(0.,100.,10000)
x = np.sin(3.1*t)
y = np.cos(11.0*t)

plt.plot(x,y)
plt.show()
