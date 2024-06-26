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
import re

def interstellar_data_prep(dataframe):

    # Değişken isimlerini düzeltme
    dataframe = dataframe.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    # Tarih değişkenlerini datetime formatına getirme
    dataframe["BookingDate"] = pd.to_datetime(dataframe["BookingDate"], format="%Y-%m-%d")
    dataframe["DepartureDate"] = pd.to_datetime(dataframe["DepartureDate"], format="%Y-%m-%d")

    # Price'ı 0'dan küçük olan satırlar var.
    dataframe = dataframe[dataframe['PriceGalacticCredits'] > 0]

    # Kategorik ve numerik değişkenleri çekme
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    # Aykırı(Outlier) değerler için gerekli işlemler
    for col in num_cols:
        if col != "CustomerSatisfactionScore":
            replace_with_thresholds(dataframe, col)

    # Eksik değerleri doldurma
    dataframe["SpecialRequests"].fillna("None", inplace=True)

    # Yeni Değişkenler Oluşturma

    # Kaldığı gece üzerinden olan verileri kaldığı gün olarak değiştirme işlemi.
    dataframe["DurationofStayEarthDays"] = dataframe["DurationofStayEarthDays"].apply(lambda x: x + 1)

    dataframe["NEW_TOTAL_DAYS_BOOKING_TO_DEPARTURE"] = dataframe["DepartureDate"] - dataframe["BookingDate"]
    dataframe["NEW_TOTAL_DAYS_BOOKING_TO_DEPARTURE"] = dataframe["NEW_TOTAL_DAYS_BOOKING_TO_DEPARTURE"].dt.days

    dataframe["NEW_COST_PRICE_TO_DISTANCE"] = dataframe["PriceGalacticCredits"] / dataframe["DistancetoDestinationLightYears"].replace(0,
                                                                                                                  0.01)
    dataframe["NEW_COST_PRICE_TO_STAY"] = dataframe["PriceGalacticCredits"] / dataframe["DurationofStayEarthDays"].replace(0, 1)
    dataframe["NEW_COST_PRICE_TO_COMPANIONS"] = dataframe["PriceGalacticCredits"] / dataframe["NumberofCompanions"].replace(0, 1)

    dataframe["DepartureYear"] = dataframe["DepartureDate"].dt.year
    dataframe["BookingYear"] = dataframe["BookingDate"].dt.year

    dataframe["Destination"] = dataframe['Destination'].apply(lambda x: 'Exotic' if 'Exotic' in x else x)

    dataframe.loc[(dataframe['Age'] < 15), 'NEW_AGE_CAT'] = "children"
    dataframe.loc[(dataframe['Age'] >= 15) & (dataframe['Age'] < 25), 'NEW_AGE_CAT'] = "young"
    dataframe.loc[(dataframe['Age'] >= 25) & (dataframe['Age'] < 40), 'NEW_AGE_CAT'] = "mature"
    dataframe.loc[(dataframe['Age'] >= 40) & (dataframe['Age'] < 65), 'NEW_AGE_CAT'] = "middle_age"
    dataframe.loc[(dataframe['Age'] >= 65), 'NEW_AGE_CAT'] = "senior"

    # Age x Gender
    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "children") & (dataframe['Gender'] == "Male"), 'NEW_AGE_GENDER'] = "children_male"
    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "children") & (dataframe['Gender'] == "Female"), 'NEW_AGE_GENDER'] = "children_female"

    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "young") & (dataframe['Gender'] == "Male"), 'NEW_AGE_GENDER'] = "young_male"
    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "young") & (dataframe['Gender'] == "Female"), 'NEW_AGE_GENDER'] = "young_female"

    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "mature") & (dataframe['Gender'] == "Male"), 'NEW_AGE_GENDER'] = "mature_male"
    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "mature") & (dataframe['Gender'] == "Female"), 'NEW_AGE_GENDER'] = "mature_female"

    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "middle_age") &
           (dataframe['Gender'] == "Male"), 'NEW_AGE_GENDER'] = "middle_age_male"
    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "middle_age") &
           (dataframe['Gender'] == "Female"), 'NEW_AGE_GENDER'] = "middle_age_female"

    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "senior") &
           (dataframe['Gender'] == "Male"), 'NEW_AGE_GENDER'] = "senior_male"
    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "senior") &
           (dataframe['Gender'] == "Female"), 'NEW_AGE_GENDER'] = "senior_female"

    # Gender x Occupation
    dataframe.loc[(dataframe['Gender'] == "Male") &
           (dataframe['Occupation'] == "Colonist"), 'NEW_OCCUPATION_GENDER'] = "male_colonist"
    dataframe.loc[(dataframe['Gender'] == "Female") &
           (dataframe['Occupation'] == "Colonist"), 'NEW_OCCUPATION_GENDER'] = "female_colonist"

    dataframe.loc[(dataframe['Gender'] == "Male") &
           (dataframe['Occupation'] == "Tourist"), 'NEW_OCCUPATION_GENDER'] = "male_tourist"
    dataframe.loc[(dataframe['Gender'] == "Female") &
           (dataframe['Occupation'] == "Tourist"), 'NEW_OCCUPATION_GENDER'] = "female_tourist"

    dataframe.loc[(dataframe['Gender'] == "Male") &
           (dataframe['Occupation'] == "Businessperson"), 'NEW_OCCUPATION_GENDER'] = "male_businessperson"
    dataframe.loc[(dataframe['Gender'] == "Female") &
           (dataframe['Occupation'] == "Businessperson"), 'NEW_OCCUPATION_GENDER'] = "female_businessperson"

    dataframe.loc[(dataframe['Gender'] == "Male") &
           (dataframe['Occupation'] == "Explorer"), 'NEW_OCCUPATION_GENDER'] = "male_explorer"
    dataframe.loc[(dataframe['Gender'] == "Female") &
           (dataframe['Occupation'] == "Explorer"), 'NEW_OCCUPATION_GENDER'] = "female_explorer"

    dataframe.loc[(dataframe['Gender'] == "Male") &
           (dataframe['Occupation'] == "Scientist"), 'NEW_OCCUPATION_GENDER'] = "male_scientist"
    dataframe.loc[(dataframe['Gender'] == "Female") &
           (dataframe['Occupation'] == "Scientist"), 'NEW_OCCUPATION_GENDER'] = "female_scientist"

    dataframe.loc[(dataframe['Gender'] == "Male") &
           (dataframe['Occupation'] == "Other"), 'NEW_OCCUPATION_GENDER'] = "male_other"
    dataframe.loc[(dataframe['Gender'] == "Female") &
           (dataframe['Occupation'] == "Other"), 'NEW_OCCUPATION_GENDER'] = "female_other"

    # TravelClass x TransportationType
    dataframe['NEW_CLASS_TRANSPORTATION'] = dataframe['TravelClass'].str.cat(dataframe['TransportationType'], sep='_')
    dataframe['NEW_CLASS_TRANSPORTATION'] = dataframe['NEW_CLASS_TRANSPORTATION'].str.lower()
    dataframe['NEW_CLASS_TRANSPORTATION'] = dataframe['NEW_CLASS_TRANSPORTATION'].str.replace(" ", "_")

    # LoyaltyProgramMember x Gender
    dataframe.loc[(dataframe['Gender'] == "Male") &
           (dataframe['LoyaltyProgramMember'] == "Yes"), 'NEW_CUSTOMER_GENDER'] = "male_loyal"
    dataframe.loc[(dataframe['Gender'] == "Female") &
           (dataframe['LoyaltyProgramMember'] == "Yes"), 'NEW_CUSTOMER_GENDER'] = "female_loyal"

    dataframe.loc[(dataframe['Gender'] == "Male") &
           (dataframe['LoyaltyProgramMember'] == "No"), 'NEW_CUSTOMER_GENDER'] = "male_unloyal"
    dataframe.loc[(dataframe['Gender'] == "Female") &
           (dataframe['LoyaltyProgramMember'] == "No"), 'NEW_CUSTOMER_GENDER'] = "female_unloyal"

    # LoyaltyProgramMember x PurposeofTravel
    dataframe.loc[(dataframe['LoyaltyProgramMember'] == "Yes") & (
            dataframe["PurposeofTravel"] == "Business"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "business_loyal"
    dataframe.loc[(dataframe['LoyaltyProgramMember'] == "Yes") & (
            dataframe["PurposeofTravel"] == "Tourism"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "tourism_loyal"
    dataframe.loc[(dataframe['LoyaltyProgramMember'] == "Yes") & (
            dataframe["PurposeofTravel"] == "Research"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "research_loyal"
    dataframe.loc[(dataframe['LoyaltyProgramMember'] == "Yes") & (
            dataframe["PurposeofTravel"] == "Colonization"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "colonization_loyal"
    dataframe.loc[(dataframe['LoyaltyProgramMember'] == "Yes") & (
            dataframe["PurposeofTravel"] == "Other"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "other_loyal"

    dataframe.loc[(dataframe['LoyaltyProgramMember'] == "No") & (
            dataframe["PurposeofTravel"] == "Business"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "business_unloyal"
    dataframe.loc[(dataframe['LoyaltyProgramMember'] == "No") & (
            dataframe["PurposeofTravel"] == "Tourism"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "tourism_unloyal"
    dataframe.loc[(dataframe['LoyaltyProgramMember'] == "No") & (
            dataframe["PurposeofTravel"] == "Research"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "research_unloyal"
    dataframe.loc[(dataframe['LoyaltyProgramMember'] == "No") & (
            dataframe["PurposeofTravel"] == "Colonization"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "colonization_unloyal"
    dataframe.loc[(dataframe['LoyaltyProgramMember'] == "No") & (
            dataframe["PurposeofTravel"] == "Other"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "other_unloyal"

    # NEW_CUSTOMER_PURPOSE_OF_TRAVEL x Gender
    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "business_loyal") & (
            dataframe['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "business_loyal_male"
    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "business_loyal") & (
            dataframe['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "business_loyal_female"

    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "tourism_loyal") & (
            dataframe['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "tourism_loyal_male"
    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "tourism_loyal") & (
            dataframe['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "tourism_loyal_female"

    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "research_loyal") & (
            dataframe['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "research_loyal_male"
    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "research_loyal") & (
            dataframe['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "research_loyal_female"

    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "colonization_loyal") & (
            dataframe['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "colonization_loyal_male"
    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "colonization_loyal") & (
            dataframe['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "colonization_loyal_female"

    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "other_loyal") & (
            dataframe['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "other_loyal_male"
    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "other_loyal") & (
            dataframe['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "other_loyal_female"

    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "business_unloyal") & (
            dataframe['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "business_unloyal_male"
    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "business_unloyal") & (
            dataframe['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "business_unloyal_female"

    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "tourism_unloyal") & (
            dataframe['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "tourism_unloyal_male"
    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "tourism_unloyal") & (
            dataframe['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "tourism_unloyal_female"

    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "research_unloyal") & (
            dataframe['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "research_unloyal_male"
    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "research_unloyal") & (
            dataframe['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "research_unloyal_female"

    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "colonization_unloyal") & (
            dataframe['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "colonization_unloyal_male"
    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "colonization_unloyal") & (
            dataframe['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "colonization_unloyal_female"

    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "other_unloyal") & (
            dataframe['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "other_unloyal_male"
    dataframe.loc[(dataframe['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "other_unloyal") & (
            dataframe['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "other_unloyal_female"

    # Son güncel değişken türlerimi tutuyorum.
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    # Encoding işlemleri
    dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)

    # Standartlaştırma
    num_cols = [col for col in num_cols if col not in ["CustomerSatisfactionScore", "BookingDate", "DepartureDate"]]
    ss = StandardScaler()
    dataframe[num_cols] = ss.fit_transform(dataframe[num_cols])

    # Bağımsız değişkenler ve hedef değişken
    y = dataframe["CustomerSatisfactionScore"]
    X = dataframe.drop(["CustomerSatisfactionScore", "StarSystem", "BookingDate", "DepartureDate"], axis=1)

    return X,y

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

# config.py

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

