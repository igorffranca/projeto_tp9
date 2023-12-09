import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

dados = pd.read_csv("HousePrices_HalfMil.csv")

X = dados[['Baths', 'Garage']]
y = dados['Prices']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

st.title('Dashboard de Regressão Linear - Análise Exploratória e Modelo')

st.subheader('Dados Originais')
st.write(dados.head())

st.subheader('Resultados da Regressão Linear')
st.write(f"Coeficientes: {modelo.coef_}")
st.write(f"Intercepto: {modelo.intercept_}")
st.write(f"MSE: {mse}")
st.write(f"RMSE: {rmse}")
st.write(f"MAE: {mae}")

st.subheader('Gráfico de Dispersão com Linha de Regressão')
scatter_fig, scatter_ax = plt.subplots()
scatter_ax.scatter(X_test['Baths'], y_test, label='Valores reais')
scatter_ax.plot(X_test['Baths'], y_pred, label='Predições', color='red', linewidth=2)
scatter_ax.set_xlabel('Baths')
scatter_ax.set_ylabel('Prices')
st.set_option('deprecation.showPyplotGlobalUse', False)
scatter_ax.legend()
st.pyplot(scatter_fig)
