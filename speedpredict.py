import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
data = pd.read_csv(r'static/data/cdspeedpredict.csv')
data = data.set_index('batch_time')
data.index = pd.to_datetime(data.index)
ts = data['speed']
decomposition = sm.tsa.seasonal_decompose(ts, model='additive',period=144,two_sided=False)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
trend = pd.DataFrame(trend)
trend.dropna(inplace=True)
'''decomposition.plot()
plt.show()'''
#自相关图平稳性检测
'''plot_acf(trend)
plot_pacf(trend)
plt.show()'''
#adf平稳性检测
'''print(ADF(ts))'''
#白噪声检测
'''print(acorr_ljungbox(trend, lags = 144,boxpierce=False))'''
#分解周期性
'''decomposition = sm.tsa.seasonal_decompose(ts, model='additive',period=144,two_sided=False)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
trend = pd.DataFrame(trend)
trend.dropna(inplace=True)
decomposition.plot()
plt.show()'''

#ARMA模型定阶
'''order = st.arma_order_select_ic(trend,max_ar=2,max_ma=2,ic=['aic','bic'])
print(order.bic_min_order)'''
#建立ARMA模型
model=ARMA(trend, (2, 0)).fit(disp=-1, method='css')
pre_trend=model.forecast(4465)[0]
pre_range=pd.date_range(start=trend.index[-1], periods=4465, freq='10T')[1:]
pre_values=[]
for i, t in enumerate(pre_range):
    trend_part =pre_trend[i]
    season_part=seasonal[seasonal.index.time==t.time()].mean()
    predict=trend_part+season_part
    pre_values.append(predict)
final_pred = pd.Series(pre_values, index=pre_range, name='predict')
'''final_pred.to_csv('static/data/speedpredict.csv')'''
final_pred.plot(color='red',legend=True,label='predict')
ts.plot()
plt.show()


