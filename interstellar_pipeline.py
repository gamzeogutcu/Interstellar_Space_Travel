import joblib
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from xgboost import XGBRegressor
from Func import *


# Base Models
def base_models(X,y):
    print("Base Models...")
    regressors = [  # ('LR', LinearRegression()),
        # ("Ridge", Ridge()),
        # ("Lasso", Lasso()),
        # ("ElasticNet", ElasticNet()),
        # ('KNN', KNeighborsRegressor()),
        ('CART', DecisionTreeRegressor()),
        # ('RF', RandomForestRegressor()),
        # ('SVR', SVR()),
        # ('GBM', GradientBoostingRegressor()),
        # ("XGBoost", XGBRegressor(objective='reg:squarederror')),
        ("LightGBM", LGBMRegressor(verbosity=-1)),
        ("CatBoost", CatBoostRegressor(verbose=False))]

    for name, regressor in regressors:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")


# Hyperparameter Optimization


knn_params={"n_neighbors":range(2,50)}

cart_params = {'max_depth': range(1, 20),
               'min_samples_split': range(2, 30)}

rf_params = {"max_depth":[8,15,None],
             "max_features":[5,7,"auto"],
             "min_samples_split":[15,20],
             "n_estimators":[200,300]}

xgboost_params = {"learning_rate":[0.1,0.2],
                  "max_depth":[None,8],
                  "n_estimators":[100,200]}

lightgbm_params = {"learning_rate":[0.01,0.1],
                  "n_estimators":[100,300],
                  "colsample_bytree":[0.5,1]}

catboost_params = {"learning_rate": [0.1, 0.5],
                   'n_estimators': [500,1500],
                   "depth": [4, 6]}


regressors=[#('KNN',KNeighborsRegressor(),knn_params),
             #("CART",DecisionTreeRegressor(),cart_params),
             #("RF",RandomForestRegressor(),rf_params),
             ("XGBoost", XGBRegressor(objective='reg:squarederror'),xgboost_params),
             ('LightGBM',LGBMRegressor(verbosity=-1),lightgbm_params),
             ('CatBoost',CatBoostRegressor(verbose=False),catboost_params)]


def hyperparameter_optimization(X, y, cv=5):
    print("Hyperparameter Optimization")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=cv, scoring="neg_mean_squared_error")))
        print(f"RMSE  (Before): {round(rmse, 4)} ({name}) ")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)

        rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=cv, scoring="neg_mean_squared_error")))
        print(f"RMSE  (After): {round(rmse, 4)} ({name}) ")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model

    return best_models


# Stacking & Ensemble Learning
def voting_regressor(best_models,X,y):
    print("Voting Regressor...")
    voting_rgs=VotingRegressor(estimators=[('CatBoost',best_models["CatBoost"]),
                                            ('LightGBM',best_models["LightGBM"]),
                                            ('XGBoost',best_models["XGBoost"])]).fit(X,y)
    rmse = np.mean(np.sqrt(-cross_val_score(voting_rgs, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)}")
    return voting_rgs


# Pipeline Main Function

def main():
    dataframe=pd.read_csv("datasets/interstellar_travel.csv")
    dataframe = dataframe.sample(n=50000, random_state=17)
    X,y=interstellar_data_prep(dataframe)
    base_models(X,y)
    best_models=hyperparameter_optimization(X,y)
    voting_rgs = voting_regressor(best_models, X, y)
    joblib.dump(voting_rgs, "voting_rgs.pkl")
    return voting_rgs


# Bir python dosyasını çalıştıracak olan nihai bölümdür.
if __name__=="__main__":
    print("İşlem başladı...")
    main()

