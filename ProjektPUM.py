import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

# Wczytanie danych treningowych i testowych
X_train = pd.read_csv('X_train_projekt.csv')
X_test = pd.read_csv('X_test_projekt.csv')
y_train = pd.read_csv('y_train_projekt.csv')
y_test = pd.read_csv('y_test_projekt.csv')

# Tworzenie i trenowanie modelu k-NN
k = 3  # wartość k
model = KNeighborsRegressor(n_neighbors=k)
model.fit(X_train, y_train)

# Przewidywanie na zbiorze testowym
y_pred = model.predict(X_test)

# Ocena modelu
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
rmse = np.sqrt(mse)
print("Root Mean Square Error (RMSE):", rmse)
rmae = np.sqrt(mae)
print("Root Mean Absolute Error (RMAE):", rmae)

# Rzeczywiste wartości i przewidywane wartości PM 2.5
plt.figure(figsize=(12, 3))  # Ustawienie rozmiaru figury
plt.plot(y_test.values, label='Rzeczywiste')
plt.plot(y_pred, label='Prognozy')

# Konfiguracja osi i tytułu wykresu
plt.xlabel('Indeks próbki')
plt.ylabel('PM 2.5')
plt.title('Prognozowane wartości PM 2.5 - metoda k-NN')

plt.legend() # Legenda

plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals) # wykres rozrzutu reszt
plt.axhline(y=0, color='r', linestyle='-') # linia pozioma o wartości 0
plt.title('Residual Plot')
plt.xlabel('Prognozowane wartości')
plt.ylabel('Residuals')
plt.show()
