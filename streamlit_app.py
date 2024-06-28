import pickle
from datetime import datetime
import numpy as np
from Func import *




def main():
    # Arka plan resmini yüklemek için dosya adını belirtin
    set_bg_hack('background_images/background.jpg')

    # sidebar için arkaplan resmi
    sidebar_bg("background_images/sidebar_background.jpg")

    # Başlık ve üstbilgi
    html_temp = """
    <h1 style='color:#ffffff;'>Customer Satisfaction Score</h1>"""
    st.markdown(html_temp, unsafe_allow_html=True)

    # Altbaşlık
    st.markdown("""
        <h2 style='color: white;margin-bottom: -30px;'>Customer Input Features</h2>
        <hr style='height: 5px; border: 0; background: linear-gradient(to right, violet, indigo, blue, green, yellow, orange, red);margin-bottom: 40px;'>
    """, unsafe_allow_html=True)

    # Sidebar with feature input
    st.sidebar.markdown("<h1 style='color: white;'>Input Features</h1>", unsafe_allow_html=True)

    # Metin ve veri ekleme

    # Yaş
    Age = st.sidebar.number_input("Age", min_value=0, max_value=120)

    # Cinsiyet
    Gender = st.sidebar.radio("Gender", ["Male", "Female"])

    # Meslek
    Occupation = st.sidebar.selectbox('Occupation',
                                      ['Tourist', 'Explorer', 'Colonist', 'Businessperson', 'Scientist', 'Other'])

    # Seyahat Sınıfı
    TravelClass = st.sidebar.selectbox('Travel Class', ['Luxury', 'Economy', 'Business'])

    # Hedef
    Destination = st.sidebar.selectbox('Destination', ["Barnard's Star",
                                                       "Proxima Centauri",
                                                       "Lalande 21185",
                                                       "Kepler-22b",
                                                       "Alpha Centauri",
                                                       "Zeta II Reticuli",
                                                       "Epsilon Eridani",
                                                       "Trappist-1",
                                                       "Gliese 581",
                                                       "Tau Ceti",
                                                       "Exotic Destination 8",
                                                       "Exotic Destination 6",
                                                       "Exotic Destination 5",
                                                       "Exotic Destination 2",
                                                       "Exotic Destination 3",
                                                       "Exotic Destination 7",
                                                       "Exotic Destination 10",
                                                       "Exotic Destination 9",
                                                       "Exotic Destination 1",
                                                       "Exotic Destination 4"])

    # Hedefe Uzaklık
    DistancetoDestination = st.sidebar.number_input("Distance to Destination (Light-Years):", format="%f",min_value=0.000001)
    # Kalış Süresi
    DurationofStay = st.sidebar.number_input("Duration of Stay (EarthDays):", min_value=0)

    # Yol Arkadaşlarının Sayısı
    NumberofCompanions = st.sidebar.slider("Number of Companions", 0, 8, step=1)

    # Seyahat Amacı
    PurposeofTravel = st.sidebar.selectbox('Purpose of Travel',
                                           ["Business", "Tourism", "Colonization", "Other", "Research"])

    # Ulaşım Türü
    TransportationType = st.sidebar.selectbox('Transportation Type',
                                              ["Solar Sailing", "Warp Drive", "Other", "Ion Thruster"])

    # Fiyat
    Price = st.sidebar.number_input("Price(GalacticCredits):", format="%f",min_value=0.000001)


    if Price <= 0:
        st.sidebar.error("Price must be greater than 0.0")

    # Rezervasyon Tarihi
    BookingDate = st.sidebar.date_input(
        "Booking Date: yyyy-mm-dd",
        min_value=datetime(2022, 1, 24),
        max_value=datetime(2024, 1, 24)
    )
    if BookingDate:
        # Tarih bilgisini stringe çevirme
        BookingDate = BookingDate.strftime("%Y-%m-%d")

    # Kalkış Tarihi
    DepartureDate = st.sidebar.date_input(
        "Departure Date: yyyy-mm-dd",
        min_value=datetime(2024, 1, 25),
        max_value=datetime(2026, 1, 23)
    )
    if DepartureDate:
        DepartureDate=DepartureDate.strftime("%Y-%m-%d")

    # Özel İstekler
    SpecialRequests = st.sidebar.selectbox('Special Requests',
                                           ["Special Meal", "Window Seat", "Other", "Extra Space Suit"])

    # Sadakat Programı Üyesi
    LoyaltyProgramMember = st.sidebar.radio("Are you a member of our loyalty program?", ("Yes", "No"))

    # Rezervasyon Ayı
    Month = st.sidebar.slider("Booking Month: ", 1, 12, step=1)

    data = {
        "Age": int(Age),
        "Gender": Gender,
        "Occupation": Occupation,
        "Travel Class": TravelClass,
        "Destination": Destination,
        "Distance to Destination (Light-Years)": float(DistancetoDestination),
        "Duration of Stay (Earth Days)": float(DurationofStay),
        "Number of Companions": int(NumberofCompanions),
        "Purpose of Travel": PurposeofTravel,
        "Transportation Type": TransportationType,
        "Price (Galactic Credits)": float(Price),
        "Booking Date": BookingDate,
        "Departure Date": DepartureDate,
        "Special Requests": SpecialRequests,
        "Loyalty Program Member": LoyaltyProgramMember,
        "Month": Month
    }

    df_new = pd.DataFrame.from_dict([data])
    st.dataframe(df_new)


    if st.sidebar.button("PREDICT"):

        df = pd.read_csv("datasets/interstellar_travel.csv")
        df = df.sample(n=50000, random_state=17)
        df = pd.concat([df, df_new], ignore_index=True)
        X, y = interstellar_data_prep(df)

        df_new = X.iloc[-1]

        df_new = [float(val) if idx < 8 else int(val) for idx, val in enumerate(df_new)]
        # Input Verilerinden Tek Satırlık DataFrame Oluşturma
        np_list = np.array(df_new).reshape(1, -1)
        new_df = pd.DataFrame(np_list)
        model = pickle.load(open("lightgbm_model.pkl", "rb"))
        prediction = model.predict(new_df)
        with st.spinner("Tahmin yapılıyor..."):
            result_message = "The estimated satisfaction score is {}.".format(prediction[0])
            st.write(
                f'<div style="background-color:#262730; padding: 10px; border-radius: 5px;"><span style="color:white;">{result_message}</span></div>',
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()

