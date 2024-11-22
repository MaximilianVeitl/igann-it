import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True, as_frame=True)
scaler = StandardScaler()
X_names = X.columns

X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=X_names)
X['sex'] = X.sex.apply(lambda x: 'w' if x > 0 else 'm')

y = (y - y.mean()) / y.std()

from igann import IGANN
model = IGANN(task='regression')
model.fit(X, y)

model.plot_single(plot_by_list=['age', 'bmi', 'bp', 'sex', 's1', 's2'])

model.predict(X)

# Regression Model
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X, y)

X1, y1 = load_diabetes(return_X_y=True, as_frame=True)
X1
print('test')
