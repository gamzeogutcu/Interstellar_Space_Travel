import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import sklearn.metrics as mt
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=FutureWarning)
from Func import *

# Features
#######################
# Age: Age of the traveler.
# Gender: Gender of the traveler.
# Occupation: Occupation of the traveler, such as Colonist, Tourist, Businessperson, etc.
# Travel Class: Class of travel, e.g., Business, Economy, Luxury.
# Destination: Interstellar destination.
# Star System: Star system of the destination.
# Distance to Destination (Light-Years): The distance to the destination measured in light-years.
# Duration of Stay (Earth Days): Duration of stay at the destination in Earth days.
# Number of Companions: The number of companions accompanying the traveler.
# Purpose of Travel: The primary purpose of travel, e.g., Tourism, Research, Colonization.
# Transportation Type: Type of transportation, e.g., Warp Drive, Solar Sailing, Ion Thruster.
# Price (Galactic Credits): Price of the trip in Galactic Credits.
# Booking Date: Date when the trip was booked.
# Departure Date: Date of departure.
# Special Requests: Any special requests made by the traveler.
# Loyalty Program Member: Indicates if the traveler is a member of a loyalty program.
# Month: Month of travel.

# Veri Setini okuma
df = pd.read_csv("datasets/interstellar_travel.csv")
df = df.sample(n=50000, random_state=17)

# Genel Resim
check_df(df)

# Değişken isimlerini düzeltme
df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

# Tarih değişkenlerini datetime formatına getirme
df["BookingDate"] = pd.to_datetime(df["BookingDate"], format="%Y-%m-%d")
df["DepartureDate"] = pd.to_datetime(df["DepartureDate"], format="%Y-%m-%d")

# Price'ı 0'dan küçük olan satırlar var.
df[0 > df['PriceGalacticCredits']].count()
df = df[df['PriceGalacticCredits'] > 0]

##############################################################
# Kategorik ve numerik değişkenleri çekme
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız.

# Kategorik değişkenler için

for col in cat_cols:
    cat_summary(df, col, True)

# Numerik değişkenler için
num_cols = [col for col in num_cols if col not in ["BookingDate", "DepartureDate"]]

for col in num_cols:
    num_summary(df, col, plot=True)

# Adım 4: Hedef değişken analizi yapınız.

for col in cat_cols:
    target_summary_with_cat(df, "CustomerSatisfactionScore", col, plot=True)

# Bağımlı değişkenin incelenmesi
sns.displot(df["CustomerSatisfactionScore"], kde=True, bins=25)
plt.show()

# Adım 5: Korelasyon analizi yapınız
corr = df[num_cols].corr(method="spearman")

# Korelasyon matrisi
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Adım 1:  Aykırı (Outlier) değerler için gerekli işlemleri yapınız.

for col in num_cols:
    if col != "CustomerSatisfactionScore":
        print(col, check_outlier(df, col))

for col in num_cols:
    if col != "CustomerSatisfactionScore":
        replace_with_thresholds(df, col)

# Adım 2:  Eksik değerler için gerekli işlemleri yapınız.
df.isnull().sum()

# Eksik verileri doldurma
df["SpecialRequests"].fillna("None", inplace=True)

# Yeni Değişkenler Oluşturma

# Kaldığı gece üzerinden olan verileri kaldığı gün olarak değiştirme işlemi.
df["DurationofStayEarthDays"] = df["DurationofStayEarthDays"].apply(lambda x: x + 1)

df["NEW_TOTAL_DAYS_BOOKING_TO_DEPARTURE"] = df["DepartureDate"] - df["BookingDate"]
df["NEW_TOTAL_DAYS_BOOKING_TO_DEPARTURE"] = df["NEW_TOTAL_DAYS_BOOKING_TO_DEPARTURE"].dt.days

df["NEW_COST_PRICE_TO_DISTANCE"] = df["PriceGalacticCredits"] / df["DistancetoDestinationLightYears"].replace(0, 0.01)
df["NEW_COST_PRICE_TO_STAY"] = df["PriceGalacticCredits"] / df["DurationofStayEarthDays"].replace(0, 1)
df["NEW_COST_PRICE_TO_COMPANIONS"] = df["PriceGalacticCredits"] / df["NumberofCompanions"].replace(0, 1)

df["DepartureYear"] = df["DepartureDate"].dt.year
df["BookingYear"] = df["BookingDate"].dt.year

df["Destination"] = df['Destination'].apply(lambda x: 'Exotic' if 'Exotic' in x else x)

df.loc[(df['Age'] < 15), 'NEW_AGE_CAT'] = "children"
df.loc[(df['Age'] >= 15) & (df['Age'] < 25), 'NEW_AGE_CAT'] = "young"
df.loc[(df['Age'] >= 25) & (df['Age'] < 40), 'NEW_AGE_CAT'] = "mature"
df.loc[(df['Age'] >= 40) & (df['Age'] < 65), 'NEW_AGE_CAT'] = "middle_age"
df.loc[(df['Age'] >= 65), 'NEW_AGE_CAT'] = "senior"

# Age x Gender
df.loc[(df['NEW_AGE_CAT'] == "children") & (df['Gender'] == "Male"), 'NEW_AGE_GENDER'] = "children_male"
df.loc[(df['NEW_AGE_CAT'] == "children") & (df['Gender'] == "Female"), 'NEW_AGE_GENDER'] = "children_female"

df.loc[(df['NEW_AGE_CAT'] == "young") & (df['Gender'] == "Male"), 'NEW_AGE_GENDER'] = "young_male"
df.loc[(df['NEW_AGE_CAT'] == "young") & (df['Gender'] == "Female"), 'NEW_AGE_GENDER'] = "young_female"

df.loc[(df['NEW_AGE_CAT'] == "mature") & (df['Gender'] == "Male"), 'NEW_AGE_GENDER'] = "mature_male"
df.loc[(df['NEW_AGE_CAT'] == "mature") & (df['Gender'] == "Female"), 'NEW_AGE_GENDER'] = "mature_female"

df.loc[(df['NEW_AGE_CAT'] == "middle_age") &
       (df['Gender'] == "Male"), 'NEW_AGE_GENDER'] = "middle_age_male"
df.loc[(df['NEW_AGE_CAT'] == "middle_age") &
       (df['Gender'] == "Female"), 'NEW_AGE_GENDER'] = "middle_age_female"

df.loc[(df['NEW_AGE_CAT'] == "senior") &
       (df['Gender'] == "Male"), 'NEW_AGE_GENDER'] = "senior_male"
df.loc[(df['NEW_AGE_CAT'] == "senior") &
       (df['Gender'] == "Female"), 'NEW_AGE_GENDER'] = "senior_female"

# Gender x Occupation
df.loc[(df['Gender'] == "Male") &
       (df['Occupation'] == "Colonist"), 'NEW_OCCUPATION_GENDER'] = "male_colonist"
df.loc[(df['Gender'] == "Female") &
       (df['Occupation'] == "Colonist"), 'NEW_OCCUPATION_GENDER'] = "female_colonist"

df.loc[(df['Gender'] == "Male") &
       (df['Occupation'] == "Tourist"), 'NEW_OCCUPATION_GENDER'] = "male_tourist"
df.loc[(df['Gender'] == "Female") &
       (df['Occupation'] == "Tourist"), 'NEW_OCCUPATION_GENDER'] = "female_tourist"

df.loc[(df['Gender'] == "Male") &
       (df['Occupation'] == "Businessperson"), 'NEW_OCCUPATION_GENDER'] = "male_businessperson"
df.loc[(df['Gender'] == "Female") &
       (df['Occupation'] == "Businessperson"), 'NEW_OCCUPATION_GENDER'] = "female_businessperson"

df.loc[(df['Gender'] == "Male") &
       (df['Occupation'] == "Explorer"), 'NEW_OCCUPATION_GENDER'] = "male_explorer"
df.loc[(df['Gender'] == "Female") &
       (df['Occupation'] == "Explorer"), 'NEW_OCCUPATION_GENDER'] = "female_explorer"

df.loc[(df['Gender'] == "Male") &
       (df['Occupation'] == "Scientist"), 'NEW_OCCUPATION_GENDER'] = "male_scientist"
df.loc[(df['Gender'] == "Female") &
       (df['Occupation'] == "Scientist"), 'NEW_OCCUPATION_GENDER'] = "female_scientist"

df.loc[(df['Gender'] == "Male") &
       (df['Occupation'] == "Other"), 'NEW_OCCUPATION_GENDER'] = "male_other"
df.loc[(df['Gender'] == "Female") &
       (df['Occupation'] == "Other"), 'NEW_OCCUPATION_GENDER'] = "female_other"

# TravelClass x TransportationType
df['NEW_CLASS_TRANSPORTATION'] = df['TravelClass'].str.cat(df['TransportationType'], sep='_')
df['NEW_CLASS_TRANSPORTATION'] = df['NEW_CLASS_TRANSPORTATION'].str.lower()
df['NEW_CLASS_TRANSPORTATION'] = df['NEW_CLASS_TRANSPORTATION'].str.replace(" ", "_")

# LoyaltyProgramMember x Gender
df.loc[(df['Gender'] == "Male") &
       (df['LoyaltyProgramMember'] == "Yes"), 'NEW_CUSTOMER_GENDER'] = "male_loyal"
df.loc[(df['Gender'] == "Female") &
       (df['LoyaltyProgramMember'] == "Yes"), 'NEW_CUSTOMER_GENDER'] = "female_loyal"

df.loc[(df['Gender'] == "Male") &
       (df['LoyaltyProgramMember'] == "No"), 'NEW_CUSTOMER_GENDER'] = "male_unloyal"
df.loc[(df['Gender'] == "Female") &
       (df['LoyaltyProgramMember'] == "No"), 'NEW_CUSTOMER_GENDER'] = "female_unloyal"

# LoyaltyProgramMember x PurposeofTravel
df.loc[(df['LoyaltyProgramMember'] == "Yes") & (
        df["PurposeofTravel"] == "Business"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "business_loyal"
df.loc[(df['LoyaltyProgramMember'] == "Yes") & (
        df["PurposeofTravel"] == "Tourism"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "tourism_loyal"
df.loc[(df['LoyaltyProgramMember'] == "Yes") & (
        df["PurposeofTravel"] == "Research"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "research_loyal"
df.loc[(df['LoyaltyProgramMember'] == "Yes") & (
        df["PurposeofTravel"] == "Colonization"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "colonization_loyal"
df.loc[(df['LoyaltyProgramMember'] == "Yes") & (
        df["PurposeofTravel"] == "Other"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "other_loyal"

df.loc[(df['LoyaltyProgramMember'] == "No") & (
        df["PurposeofTravel"] == "Business"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "business_unloyal"
df.loc[(df['LoyaltyProgramMember'] == "No") & (
        df["PurposeofTravel"] == "Tourism"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "tourism_unloyal"
df.loc[(df['LoyaltyProgramMember'] == "No") & (
        df["PurposeofTravel"] == "Research"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "research_unloyal"
df.loc[(df['LoyaltyProgramMember'] == "No") & (
        df["PurposeofTravel"] == "Colonization"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "colonization_unloyal"
df.loc[(df['LoyaltyProgramMember'] == "No") & (
        df["PurposeofTravel"] == "Other"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] = "other_unloyal"

# NEW_CUSTOMER_PURPOSE_OF_TRAVEL x Gender
df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "business_loyal") & (
            df['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "business_loyal_male"
df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "business_loyal") & (
            df['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "business_loyal_female"

df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "tourism_loyal") & (
            df['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "tourism_loyal_male"
df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "tourism_loyal") & (
            df['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "tourism_loyal_female"

df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "research_loyal") & (
            df['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "research_loyal_male"
df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "research_loyal") & (
            df['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "research_loyal_female"

df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "colonization_loyal") & (
            df['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "colonization_loyal_male"
df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "colonization_loyal") & (
            df['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "colonization_loyal_female"

df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "other_loyal") & (
            df['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "other_loyal_male"
df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "other_loyal") & (
            df['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "other_loyal_female"

df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "business_unloyal") & (
            df['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "business_unloyal_male"
df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "business_unloyal") & (
            df['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "business_unloyal_female"

df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "tourism_unloyal") & (
            df['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "tourism_unloyal_male"
df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "tourism_unloyal") & (
            df['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "tourism_unloyal_female"

df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "research_unloyal") & (
            df['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "research_unloyal_male"
df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "research_unloyal") & (
            df['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "research_unloyal_female"

df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "colonization_unloyal") & (
            df['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "colonization_unloyal_male"
df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "colonization_unloyal") & (
            df['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "colonization_unloyal_female"

df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "other_unloyal") & (
            df['Gender'] == "Male"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "other_unloyal_male"
df.loc[(df['NEW_CUSTOMER_PURPOSE_OF_TRAVEL'] == "other_unloyal") & (
            df['Gender'] == "Female"), 'NEW_CUSTOMER_PURPOSE_OF_TRAVEL_GENDER'] = "other_unloyal_female"

# Adım 3:  Encoding işlemlerini gerçekleştiriniz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.shape

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
num_cols = [col for col in num_cols if col not in ["CustomerSatisfactionScore", "BookingDate", "DepartureDate"]]
ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])

# Adım 4: Model oluşturunuz.
# Bağımsız değişkenler ve hedef değişken
y = df["CustomerSatisfactionScore"]
X = df.drop(["CustomerSatisfactionScore", "StarSystem", "BookingDate", "DepartureDate"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Modelleri oluşturma
models = [  # ('LR', LinearRegression()),
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

# Her bir model için performansı değerlendirme
for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

######################################################
#  Automated Hyperparameter Optimization
######################################################
cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500]}

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

xgboost_params = {
    'learning_rate': [0.1, 0.2],
    'max_depth': [None, 5],
    'n_estimators': [100, 200]
}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

catboost_params = {"learning_rate": [0.1, 0.5],
                   'n_estimators': [500,1500],
                   "depth": [4, 6]}


regressors = [("CART", DecisionTreeRegressor(), cart_params),
              ("RF", RandomForestRegressor(), rf_params),
              # ('GBM', GradientBoostingRegressor(), gbm_params),
              ("CatBoost", CatBoostRegressor(verbose=False), catboost_params),
              ('LightGBM', LGBMRegressor(verbosity=-1), lightgbm_params)]

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

###########################################################################
lightgbm_model = LGBMRegressor(verbosity=-1).fit(X_train, y_train)
rmse = np.mean(np.sqrt(-cross_val_score(lightgbm_model, X, y, cv=10, scoring="neg_mean_squared_error")))
print(f"Lightgbm Model RMSE: {round(rmse, 4)} ")

y_pred = lightgbm_model.predict(X_test)
r2 = mt.r2_score(y_test, y_pred)
print(f"LightGBM Model R2: {round(r2, 4)} ")

# LightGBM Hiperparametre optimizasyonu

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

grid_search = GridSearchCV(lightgbm_model, lightgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
lightgbm_final = lightgbm_model.set_params(**grid_search.best_params_)
rmse = np.mean(np.sqrt(-cross_val_score(lightgbm_final, X_test, y_test, cv=10, scoring="neg_mean_squared_error")))
print(f"Lightgbm Final Model RMSE: {round(rmse, 4)} ")


###########################################################################
def plot_importance(model, features, num=20, save=False):
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError(
            f"The model does not have feature importances. Ensure that the model is fitted and has the attribute 'feature_importances_'.")

    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})

    if num > len(features.columns):
        num = len(features.columns)

    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:num])
    plt.title('Feature Importances')
    plt.tight_layout()

    if save:
        plt.savefig('importances.png')
    plt.show()


plot_importance(lightgbm_model, X_train)

#################################################
# Hata Grafiği Oluşturma

hatalar = y_test - y_pred

plt.scatter(y_pred, hatalar, color="Red")
plt.xlabel('Tahmin Edilen Değerler')
plt.ylabel('Hatalar')
plt.title('Hata Çizimi')
plt.axhline(0, color="Blue", linestyle="--")
plt.show()
########
