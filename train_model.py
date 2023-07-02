from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import joblib


filtered_data = np.load('./macroeconomic_data/filtered_data.npy', allow_pickle=True).item()

data = []
for date, data_frame in filtered_data.items():
    flattened_data = pd.json_normalize(data_frame, sep=".")
    flattened_data["10"] = str(date[1])
    data.append(flattened_data)
data = pd.concat(data, ignore_index=True)

X = data
X = X.iloc[:-1, :]
y = data.filter(regex='0.36', axis=1)
y = y.iloc[1:]
y = y.values.ravel()

random_state = np.random.RandomState()
random_state_tuple = random_state.get_state()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

base_estimator = MLPRegressor()
parameters = {
    'hidden_layer_sizes': [(200, 200, 200, 200, 200), (200, 200, 200, 200), (100, 200, 200, 100), (200, 200, 200)],
    "activation": ['logistic', 'tanh'],
    "solver": ['sgd'],
    "alpha": [0.001, 0.0001, 0.00005],
    "learning_rate": ['invscaling', 'adaptive'],
    "max_iter": [100000],
    "tol": [1e-6],
    "momentum": [0.8, 0.85, 0.9, 0.95],
    "nesterovs_momentum": [True, False]
}

clf = GridSearchCV(base_estimator, param_grid=parameters, cv=5, error_score='raise', n_jobs=-1, verbose=2,
                   scoring=make_scorer(mean_squared_error, greater_is_better=False))
clf.fit(X_train, y_train)

best_estimator = clf.best_estimator_

joblib.dump(best_estimator, './joblib/best_estimator.joblib', 3)
joblib.dump(random_state_tuple, './joblib/split_random_state.joblib', 3)
