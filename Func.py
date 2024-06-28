import base64

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import re

def check_df(dataframe, head=5):
    print("##### Shape #####")
    print(dataframe.shape)
    print("##### Types #####")
    print(dataframe.dtypes)
    print("##### Head #####")
    print(dataframe.head())
    print("##### Tail #####")
    print(dataframe.tail())
    print("##### NA #####")
    print(dataframe.isnull().sum())
    print("##### Quantiles #####")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def grab_col_names(dataframe, cat_th=15, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.xticks(rotation=45)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.xticks(rotation=45)
        plt.show(block=True)

def target_summary_with_cat(dataframe,target,categorical_col, plot=False):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))
    if plot:
        sns.violinplot(data=dataframe, x=categorical_col,y=target)
        plt.xticks(rotation=45)
        plt.show(block=True)

def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def plot_importance(model, features, num=20, save=False):
    # Ensure the model has the feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError(f"The model does not have feature importances. Ensure that the model is fitted and has the attribute 'feature_importances_'.")

    # Extract feature importances
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})

    # If num exceeds the number of features, plot all features
    if num > len(features.columns):
        num = len(features.columns)

    # Create the plot
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:num])
    plt.title('Feature Importances')
    plt.tight_layout()

    # Save the plot if specified
    if save:
        plt.savefig('importances.png')

    # Show the plot
    plt.show()

# Veri setini modele sokmadan önce yapılması gereken ön işlemler (EDA vb.)
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

##### Streamlit için kullanılan fonksiyonlar ###


# Arka plan resmini eklemek için HTML kullanımı
def set_bg_hack(main_bg, width='100%', height='100%'):
    '''
    A function to unpack an image from root folder and set as bg.

    Parameters
    ----------
    main_bg : str
        The filename of the background image.
    width : str, optional
        The width of the background image. Defaults to '100%'.
    height : str, optional
        The height of the background image. Defaults to 'auto'.

    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover;
             background-position: center;
             background-repeat: no-repeat;
             width: {width};
             height: {height};
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

# Sidebar arkaplan resmini yüklemek için
def sidebar_bg(side_bg):
    side_bg_ext = 'png'

    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] > div:first-child {{
            background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
            background-size: contain; /* Resmin orijinal boyutunu korur */
            background-repeat: no-repeat; /* Resmin tek sefer yüklenmesini sağlar */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )