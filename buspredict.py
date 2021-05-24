import inline as inline
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima_model import ARMA


plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False


data=pd.read_csv(r'static/data/buspassager.csv')
df=pd.Series(np.array(data['passager']),index=pd.period_range('20190701','20190916',freq='D'))
df.plot(legend='passengers')
plt.show()
#平稳性处理
rol_mean = df.rolling(window=2).mean()
rol_mean.dropna(inplace=True)
df_role_mean_diff1 = rol_mean.diff(1)
df_role_mean_diff1.dropna(inplace=True)
df_role_mean_diff2 = df_role_mean_diff1.diff(1)
df_role_mean_diff2.dropna(inplace=True)
#平稳性检测
'''print(ADF(df_role_mean_diff2))
print(acorr_ljungbox(df_role_mean_diff2, lags = [6, 12],boxpierce=False))'''
#ARMA模型定阶
'''order = st.arma_order_select_ic(df_role_mean_diff2,max_ar=5,max_ma=3,ic=['aic','bic'])
print(order.bic_min_order)'''
#建立ARMA模型
model = ARMA(df_role_mean_diff2, order=(4,2))
result_arma = model.fit(disp=-1, method='css')
'''predict_ts = result_arma.predict()
#一阶差分还原
diff_shift_ts = df_role_mean_diff1.shift(1)
diff_recover_1 = predict_ts.add(diff_shift_ts)
#再次一阶差分还原
rol_shift_ts = rol_mean.shift(1)
diff_recover = diff_recover_1.add(rol_shift_ts)
#移动平均还原
rol_sum = df.rolling(window=1).sum()
rol_recover = diff_recover*2 - rol_sum.shift(1)
#画出拟合值和实际值
df.plot(figsize=(10,5))
rol_recover.plot(color='red')
plt.show()'''
#预测后9天的客运量
forecast=result_arma.forecast(7)
forecast=pd.Series(forecast[0],index=pd.period_range('20190917','20190923',freq='D'))
#一二阶差分还原
diff_1=pd.concat([df_role_mean_diff1.iloc[-1:],forecast]).cumsum()
diff_2=pd.concat([rol_mean.iloc[-1:],diff_1[1:]]).cumsum()
#移动平均还原
role=pd.concat([rol_mean,diff_2[1:]])
rol_sum = role.rolling(window=1).sum()
rol_recover = role*2 - rol_sum.shift(1)

#画出预测值
df.plot(figsize=(10,5),label='Original')
rol_recover.plot(color='red', label='Predict')
plt.legend(loc='best')
plt.show()
#输出为csv文件
'''rol_recover.to_csv('static/data/buspredict.csv')'''
