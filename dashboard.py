import streamlit as st
from datetime import date
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
import scipy.optimize as solver


tw100=pd.read_csv("https://raw.githubusercontent.com/antony0315/stockprofolio/main/TW100.csv?token=GHSAT0AAAAAABUK4YUE634NGOMPMFMLG332YTXZV5A",header=None)

st.title("效率前緣資產組合配置")
st.markdown("作者:Antony JHU")
selected_stocks=st.multiselect("select stock",options=list(tw100[0]))
st.write(selected_stocks)
select=np.array(selected_stocks)

start='2015-01-01'
today=date.today().strftime("%Y-%m-%d")

df1=pd.DataFrame()
for i in select:
    stock=str(i)+'.TW'
    price=web.DataReader(stock,"yahoo",start,today)['Adj Close']
    price=pd.DataFrame({stock:price})
    df1=pd.concat([df1,price],axis='columns')
df1=df1.fillna(method='ffill')
st.write('股價表現')
st.dataframe(df1)
fig, ax = plt.subplots(dpi=800)
heatmap=sns.heatmap(df1.corr(), vmin=-1, vmax=1, annot=True)
st.pyplot(fig)

total_stocks = len(df1.columns)
returns = df1.pct_change()
returns = returns[1:]
covariance_matrix = returns.cov() * 252
stocks_expected_return = returns.mean() * 252
stocks_weights = np.array([.1,]*total_stocks )
portfolio_return = sum(stocks_weights * stocks_expected_return)
portfolio_risk = np.sqrt(reduce(np.dot, [stocks_weights, covariance_matrix, stocks_weights.T]))
st.write('預期報酬率為: '+ str(round(portfolio_return,4)))
st.write('風險為: ' + str(round(portfolio_risk,4)))


risk_list = []
return_list = []
simulations_target = 10**4
 
for _ in range(simulations_target):
 
    # random weighted
    weight = np.random.rand(total_stocks)
    weight = weight / sum(weight)
    # calculate result
    ret = sum(stocks_expected_return * weight)
    risk = np.sqrt(reduce(np.dot, [weight, covariance_matrix, weight.T]))
 
    # record
    return_list.append(ret)
    risk_list.append(risk)
 
fig = plt.figure(figsize = (10,6))
fig.suptitle('Stochastic Simulations', fontsize=18, fontweight='bold')
 
ax = fig.add_subplot()
ax.plot(risk_list, return_list, 'o')
ax.set_title(f'n={simulations_target}', fontsize=16)
ax.set_xlabel('Risk')
ax.set_ylabel('Return')
st.pyplot(fig)

def standard_deviation(weights):
    return np.sqrt(reduce(np.dot, [weights, covariance_matrix, weights.T]))


x0 = stocks_weights
bounds = tuple((0, 1) for x in range(total_stocks))
constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
minimize_variance = solver.minimize(standard_deviation, x0=x0, constraints=constraints, bounds=bounds)

mvp_risk = minimize_variance.fun
mvp_return = sum(minimize_variance.x * stocks_expected_return)

st.write('風險最小化投資組合預期報酬率為:' + str(round(mvp_return,2)))
st.write('風險最小化投資組合風險為:' + str(round(mvp_risk,2)))

for i in range(total_stocks):
    stock_symbol = str(df1.columns[i])
    weighted = str(format(minimize_variance.x[i], '.4f'))
    st.write(f'{stock_symbol} 佔投資組合權重 : {weighted}')
