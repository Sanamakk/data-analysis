import pandas as pd
from pandas_profiling import ProfileReport 
import datetime as dt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

vendas = pd.read_excel('base_vendas.xlsx')
grupos = pd.read_excel('verifica_base.xlsx')
vendas = pd.merge(
    vendas,
    grupos,
    how='left',
    left_on = 'product_category_name', right_on = 'categoria'
)
vendas = vendas.drop(['product_category_name','categoria'],axis=1)
vendas.info()
vendas.isnull().sum()
vendas = vendas[vendas.price.notnull()]
vendas = vendas[vendas.order_status != 'canceled']
vendas = vendas[vendas.order_purchase_timestamp != dt.datetime(2017,11,24)]
vendas = vendas[vendas.order_purchase_timestamp != dt.datetime(2017,11,25)]
vendas.loc[vendas.grupo.isnull(),'grupo'] = 'outros'
vendas.isnull().sum()
vendas[vendas.order_approved_at.isnull()]
vendas = vendas.drop('order_approved_at',axis=1)
vendas.groupby('order_purchase_timestamp')['price'].sum().plot.box();
vendas.head(3)
vendas.groupby('order_purchase_timestamp')['price'].sum().plot();
venda_vlr = vendas.groupby('order_purchase_timestamp')['price'].sum()
fig, ax = plt.subplots(figsize=(12,6))

x = venda_vlr.index
y = venda_vlr.values

ax.plot(x, y, linewidth=2.0)

plt.show()

venda_vlr.index[-1]

fig, ax = plt.subplots(figsize=(12,6))

x = venda_vlr.index
y = venda_vlr.values

ax.plot(x, y, linewidth=2.0)

x_reta = [venda_vlr.index[0],venda_vlr.index[-1]]
y_reta = [5000,35000]

ax.plot(x_reta, y_reta,'--r')

plt.show()

venda_vlr = venda_vlr.reset_index()
venda_vlr.info()

venda_vlr.tail(3)

venda_vlr['ajuste_data'] = (venda_vlr.order_purchase_timestamp - venda_vlr.order_purchase_timestamp.min()).dt.days

venda_vlr['ajuste_data'] = venda_vlr['ajuste_data']/venda_vlr['ajuste_data'].max()

X = venda_vlr.ajuste_data.values.reshape(-1,1)
y = venda_vlr.price

reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_
reg.intercept_
venda_vlr.head(3)
fig, ax = plt.subplots(figsize=(12,6))

x = venda_vlr.ajuste_data
y = venda_vlr.price

ax.plot(x, y, linewidth=2.0)

x_reta = venda_vlr.ajuste_data
y_reta = x_reta*reg.coef_[0] + reg.intercept_

ax.plot(x_reta, y_reta,'--r')

plt.show()

vendas.head(3)
venda_grupos = vendas.groupby(['order_purchase_timestamp','grupo'])['price'].sum().reset_index()

venda_grupos.loc[venda_grupos.grupo == 'carro','price'].plot();

venda_grupos = vendas.groupby(['order_purchase_timestamp','grupo'])['price'].sum().reset_index()

venda_grupos.order_purchase_timestamp.max()

venda_grupos.shape

treino = venda_grupos[venda_grupos.order_purchase_timestamp <= dt.datetime(2018,3,1)]
teste = venda_grupos[venda_grupos.order_purchase_timestamp > dt.datetime(2018,3,1)]

print(treino.shape)

print(teste.shape)

print(teste.shape[0]/venda_grupos.shape[0])

fig, ax = plt.subplots(figsize=(12,6))

filtro = 'beleza'

base_treino = treino[treino.grupo == filtro]
base_teste = teste[teste.grupo == filtro]

ax.plot(base_treino.order_purchase_timestamp, base_treino.price)
ax.plot(base_teste.order_purchase_timestamp, base_teste.price,'--r')

plt.show()

treino.head(3)

#X_treino = treino[['order_purchase_timestamp','grupo']]
#y_treino = treino.price

#reg = LinearRegression().fit(X_treino, y_treino)

#venda_grupos['ajuste_data'] = (venda_grupos.order_purchase_timestamp - venda_grupos.order_purchase_timestamp.min()).dt.days
#venda_grupos['ajuste_data'] = venda_grupos['ajuste_data']/venda_grupos['ajuste_data'].max()

#treino = venda_grupos[venda_grupos.order_purchase_timestamp <= dt.datetime(2018,3,1)]
#teste = venda_grupos[venda_grupos.order_purchase_timestamp > dt.datetime(2018,3,1)]

#X_treino = treino[['ajuste_data','grupo']]
#y_treino = treino.price

#reg = LinearRegression().fit(X_treino, y_treino)

treino = pd.concat([treino,pd.get_dummies(treino.grupo)],axis=1)

X_treino = treino.drop(['order_purchase_timestamp','grupo','price'],axis=1)
y_treino = treino.price

reg = LinearRegression().fit(X_treino, y_treino)

reg.score(X_treino, y_treino)

reg.coef_

reg.intercept_

treino['previsao'] = reg.predict(X_treino)

fig, ax = plt.subplots(figsize=(12,6))

filtro = 'beleza'

base_treino = treino[treino.grupo == filtro]
base_teste = teste[teste.grupo == filtro]

ax.plot(base_treino.order_purchase_timestamp, base_treino.price)
ax.plot(base_teste.order_purchase_timestamp, base_teste.price)
ax.plot(base_treino.order_purchase_timestamp,base_treino.previsao,'--r')

plt.show()

teste.head(3)

teste = pd.concat([teste,pd.get_dummies(teste.grupo)],axis=1)

X_teste = teste.drop(['order_purchase_timestamp','grupo','price'],axis=1)
y_teste = teste.price

teste['previsao'] = reg.predict(X_teste)

fig, ax = plt.subplots(figsize=(12,6))

filtro = 'beleza'

base_treino = treino[treino.grupo == filtro]
base_teste = teste[teste.grupo == filtro]

ax.plot(base_treino.order_purchase_timestamp, base_treino.price)
ax.plot(base_teste.order_purchase_timestamp, base_teste.price)
ax.plot(base_treino.order_purchase_timestamp,base_treino.previsao,'--r')
ax.plot(base_teste.order_purchase_timestamp,base_teste.previsao,'--r')

plt.show()

