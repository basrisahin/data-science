# Doğrusal Olmayan Regresyon Modelleri
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# no need to clean data again. Lets call the pickles.
train_df = pd.read_pickle(
    '/Users/basrisahin/Documents/GitHub/data-science/house_price_prediction/data/processed/train_df.pkl')
test_df = pd.read_pickle(
    '/Users/basrisahin/Documents/GitHub/data-science/house_price_prediction/data/processed/test_df.pkl')

# remove index and Id columns. no need them.
all_data = [train_df, test_df]
drop_list = ['index', 'Id']

for data in all_data:
    data.drop(drop_list, axis=1, inplace=True)

# Split train and test
X = train_df.drop('SalePrice', axis=1)
y = np.ravel(train_df[['SalePrice']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
y_train = np.ravel(y_train)

# Lets apply KNN
knn_model = KNeighborsRegressor().fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Lets change k values and see how results are changed
for k in range(20):
    k = k + 1
    knn_model = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("k:", k, "için RMSE değeri ", rmse)

# Lets apply GridSearchCV to find optimum k value
knn_params = {"n_neighbors": np.arange(2, 30, 1)}
knn_model = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn_model, knn_params, cv=10).fit(X_train, y_train)
knn_cv_model.best_params_

knn_tuned = KNeighborsRegressor(**knn_cv_model.best_params_).fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Lets apply Linear SVR
svr_model = SVR('linear').fit(X_train, y_train)
y_pred = svr_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# SVR tuning
svr_model = SVR('linear')
svr_params = {"C": [0.01, 0.001, 0.2, 0.1, 0.5, 0.8, 0.9, 1, 10, 50, 100, 200, 300, 500, 1000, 1500, 2000]}
svr_cv_model = GridSearchCV(svr_model, svr_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
svr_cv_model.best_params_

svr_tuned = SVR('linear', **svr_cv_model.best_params_).fit(X_train, y_train)
y_pred = svr_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Lets apply NON-Linear SVR
svr_model = SVR()
svr_params = {"C": [0.01, 0.001, 0.2, 0.1, 0.5, 0.8, 0.9, 1, 10, 50, 100, 200, 300, 500, 1000, 1500, 2000]}
svr_cv_model = GridSearchCV(svr_model, svr_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
svr_cv_model.best_params_

# Final Model
svr_tuned = SVR(**svr_cv_model.best_params_).fit(X_train, y_train)
y_pred = svr_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Lets apply MLP Multi Layer Perceptron
mlp_model = MLPRegressor().fit(X_train, y_train)
y_pred = mlp_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Model Tuning
mlp_params = {"alpha": [0.1, 0.01],
              "hidden_layer_sizes": [(10, 20), (5, 5)]}

mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv=5, verbose=2, n_jobs=-1).fit(X_train, y_train)
mlp_cv_model.best_params_

# Final Model
mlp_tuned = MLPRegressor(**mlp_cv_model.best_params_).fit(X_train, y_train)
y_pred = mlp_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Lets apply CART
cart_model = DecisionTreeRegressor(random_state=52)
cart_model.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Model Tuning

cart_params = {"max_depth": [2, 3, 4, 5, 10, 20, 100, 1000],
               "min_samples_split": [2, 10, 5, 30, 50, 10],
               "criterion": ["mse", "friedman_mse", "mae"]}

cart_model = DecisionTreeRegressor()
cart_cv_model = GridSearchCV(cart_model, cart_params, verbose=2, n_jobs=-1, cv=10).fit(X_train, y_train)
cart_cv_model.best_params_

cart_tuned = DecisionTreeRegressor(**cart_cv_model.best_params_).fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# How to print decision rules of a tree
# !pip install skompiler
# pip install astor
from skompiler import skompile

print(skompile(cart_tuned.predict).to('python/code'))

# Random Forests
rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Model Tuning
rf_params = {"max_depth": [5, 8, None],
             "max_features": [20, 50, 100],
             "n_estimators": [200, 500],
             "min_samples_split": [2, 5, 10]}

# rf_params2 = {"max_depth": [3, 5, 8, 10, 15, None],
#            "max_features": [5, 10, 15, 20, 50, 100],
#            "n_estimators": [200, 500, 1000],
#            "min_samples_split": [2, 5, 10, 20, 30, 50]}

rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
rf_cv_model.best_params_

rf_tuned = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Feature importance

os.getcwd()

rf_tuned.feature_importances_
Importance = pd.DataFrame({'Importance': rf_tuned.feature_importances_ * 100,
                           'Feature': X_train.columns})

plt.figure(figsize=(10, 30))
sns.barplot(x="Importance", y="Feature", data=Importance.sort_values(by="Importance", ascending=False))
plt.title('Feature Importance ')
plt.show()
plt.savefig('rf_importance.png')


# GBM
gbm_model = GradientBoostingRegressor().fit(X_train, y_train)
y_pred = gbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# # Model Tuning
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

# gbm_params2 = {"learning_rate": [0.001, 0.1, 0.01, 0.05],
#               "max_depth": [3, 5, 8, 10,20,30],
#               "n_estimators": [200, 500, 1000, 1500, 5000],
#               "subsample": [1, 0.4, 0.5, 0.7],
#               "loss": ["ls", "lad", "quantile"]}

gbm_model = GradientBoostingRegressor()
gbm_cv_model = GridSearchCV(gbm_model, gbm_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
gbm_cv_model.best_params_

gbm_tuned = GradientBoostingRegressor(**gbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = gbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# Feature Importance
Importance = pd.DataFrame({'Importance': gbm_tuned.feature_importances_ * 100,
                           'Feature': X_train.columns})

plt.figure(figsize=(10, 30))
sns.barplot(x="Importance", y="Feature", data=Importance.sort_values(by="Importance", ascending=False))
plt.title('Feature Importance ')
plt.show()
plt.savefig('gbm_importance.png')

# XGBoost
xgb = XGBRegressor().fit(X_train, y_train)
y_pred = xgb.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Model Tuning
xgb_params = {"learning_rate": [0.1, 0.01],
              "max_depth": [5, 8],
              "n_estimators": [100, 1000],
              "colsample_bytree": [0.7, 1]}

# xgb_params2 = {"learning_rate": [0.1, 0.01, 0.5],
#              "max_depth": [5, 8, 15, 20],
#              "n_estimators": [100, 200, 500, 1000],
#              "colsample_bytree": [0.4, 0.7, 1]}

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
xgb_cv_model.best_params_

# Final Model
xgb_tuned = XGBRegressor(**xgb_cv_model.best_params_).fit(X_train, y_train)
y_pred = xgb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Feature Importance
Importance = pd.DataFrame({'Importance': xgb_tuned.feature_importances_ * 100,
                           'Feature': X_train.columns})

plt.figure(figsize=(10, 30))
sns.barplot(x="Importance", y="Feature", data=Importance.sort_values(by="Importance", ascending=False))
plt.title('Feature Importance ')
plt.show()
plt.savefig('xgb_importances.png')

# LightGBM
lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Model Tuning
lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.01, 0.001, 0.1, 0.5, 1],
               "n_estimators": [200, 500, 1000, 5000],
               "max_depth": [6, 8, 10, 15, 20],
               "colsample_bytree": [1, 0.8, 0.5, 0.4]}

lgbm_cv_model = GridSearchCV(lgb_model, lgbm_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)

lgbm_cv_model.best_params_

# Final Model
lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# CatBoost
catb_model = CatBoostRegressor(verbose=False).fit(X_train, y_train)
y_pred = catb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Model Tuning
catb_params = {"iterations": [200, 500],
               "learning_rate": [0.01, 0.1],
               "depth": [3, 6]}

catb_model = CatBoostRegressor()
catb_cv_model = GridSearchCV(catb_model,
                             catb_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

catb_cv_model.best_params_

# Final Model
catb_tuned = CatBoostRegressor(**catb_cv_model.best_params_).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, y_pred))