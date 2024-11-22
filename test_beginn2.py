import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True, as_frame=True)

scaler = StandardScaler()
X_names = X.columns

X = scaler.fit_transform(X)
y = (y - y.mean()) / y.std()

X = pd.DataFrame(X, columns=X_names)

# Linear regression model
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X, y)
y_pred = linReg.predict(X)
# mse
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, y_pred)
print(f'MSE: {mse}')

# IGANN model
X['sex'] = X.sex.apply(lambda x: 'w' if x > 0 else 'm')
from igann import IGANN
model = IGANN(task='regression')
model.fit(X, y)
y_pred_igann = model.predict(X)
# mse
mse_igann = mean_squared_error(y, y_pred_igann)
print(f'MSE IGANN: {mse_igann}')