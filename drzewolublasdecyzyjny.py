import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn.model_selection import cross_val_score


X_test = pd.read_csv("C:/Users/kuba4/PycharmProjects/PUMprojektLasso/X_test_PRSA_data_2010.1.1-2014.12.csv")
X_train = pd.read_csv("C:/Users/kuba4/PycharmProjects/PUMprojektLasso/X_train_PRSA_data_2010.1.1-2014.12.csv")
y_test = pd.read_csv("C:/Users/kuba4/PycharmProjects/PUMprojektLasso/y_test_PRSA_data_2010.1.1-2014.12.csv")
y_train = pd.read_csv("C:/Users/kuba4/PycharmProjects/PUMprojektLasso/y_train_PRSA_data_2010.1.1-2014.12.csv")
print(X_test.shape)
print(X_test.info())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train.values.ravel()) #vaues.ravel - konwersja do macierzy 1d
y_forest = rf_regressor.predict(X_test)


#wykres
plt.figure(figsize=(10, 20))
plt.plot(y_test.values, label='Rzeczywiste')
plt.plot(y_forest, label='Prognozy')

plt.xlabel('Indeks próbki')
plt.ylabel('PM 2.5')
plt.title('Prognozowane wartości PM 2.5 - las losowy')

plt.legend()
plt.show()

#cross valuation?
#scores = cross_val_score(rf_regressor, X_train, y_train.values.ravel(), scoring='neg_mean_squared_error', cv=3)
#average_score = scores.mean()

#print("Cross-Validation ", scores)
#print("Average Score:", average_score)

# MSE i MAE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
mse = mean_squared_error(y_test, y_forest)
rmse = np.sqrt(mse)
print("Mean Squared Error:", rmse)
mae = mean_absolute_error(y_test, y_forest)
rmae = np.sqrt(mae)
print("Mean Absolute Error:", rmae)


#reszty
print(y_test.shape)
print(y_forest.shape)

y_forest_reshaped = np.reshape(y_forest, (-1, 1))
residuals = y_test - y_forest_reshaped
plt.scatter(y_forest, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

#importance = rf_regressor.coef_
#plt.bar(X.columns, importance)
#plt.title('Feature Importance')
#plt.xlabel('Features')
#plt.ylabel('Importance')
#plt.xticks(rotation='vertical')
#plt.show()