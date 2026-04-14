import streamlit as st
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.joblib")
    model = xgb.XGBRegressor()
    model.load_model('xgb_car_predict_model.json')
    return scaler, model

def preprocess(year, milage, manu, cyl, fuel, tit, trans, c_type, condition, scaler):
       result = pd.DataFrame(columns=['year', 'cylinders', 'odometer', 'manufacturer_acura',
              'manufacturer_alfa-romeo', 'manufacturer_aston-martin',
              'manufacturer_audi', 'manufacturer_bmw', 'manufacturer_buick',
              'manufacturer_cadillac', 'manufacturer_chevrolet',
              'manufacturer_chrysler', 'manufacturer_datsun', 'manufacturer_dodge',
              'manufacturer_ferrari', 'manufacturer_fiat', 'manufacturer_ford',
              'manufacturer_gmc', 'manufacturer_harley-davidson',
              'manufacturer_honda', 'manufacturer_hyundai', 'manufacturer_infiniti',
              'manufacturer_jaguar', 'manufacturer_jeep', 'manufacturer_kia',
              'manufacturer_land rover', 'manufacturer_lexus', 'manufacturer_lincoln',
              'manufacturer_mazda', 'manufacturer_mercedes-benz',
              'manufacturer_mercury', 'manufacturer_mini', 'manufacturer_mitsubishi',
              'manufacturer_morgan', 'manufacturer_nissan', 'manufacturer_pontiac',
              'manufacturer_porsche', 'manufacturer_ram', 'manufacturer_rover',
              'manufacturer_saturn', 'manufacturer_subaru', 'manufacturer_tesla',
              'manufacturer_toyota', 'manufacturer_volkswagen', 'manufacturer_volvo',
              'fuel_diesel', 'fuel_electric', 'fuel_gas', 'fuel_hybrid', 'fuel_other',
              'title_status_clean', 'title_status_lien', 'title_status_missing',
              'title_status_parts only', 'title_status_rebuilt',
              'title_status_salvage', 'transmission_automatic', 'transmission_manual',
              'transmission_other', 'type_SUV', 'type_bus', 'type_convertible',
              'type_coupe', 'type_hatchback', 'type_mini-van', 'type_offroad',
              'type_other', 'type_pickup', 'type_sedan', 'type_truck', 'type_van',
              'type_wagon', 'condition_excellent', 'condition_fair', 'condition_good',
              'condition_like new', 'condition_new', 'condition_salvage']
       )
       result.loc[0, 'year'] = year
       result.loc[0, 'odometer'] = milage
       result.loc[0, manu] = True
       result.loc[0, 'cylinders'] = cyl
       result.loc[0, fuel] = True
       result.loc[0, tit] = True
       result.loc[0, trans] = True
       result.loc[0, c_type] = True
       result.loc[0, condition] = True
       result.fillna(False, inplace=True)

       result = scaler.transform(result)
       return result

### Enums
manufactuers = [
        'manufacturer_acura', 'manufacturer_alfa-romeo',
       'manufacturer_aston-martin', 'manufacturer_audi', 'manufacturer_bmw',
       'manufacturer_buick', 'manufacturer_cadillac', 'manufacturer_chevrolet',
       'manufacturer_chrysler', 'manufacturer_datsun', 'manufacturer_dodge',
       'manufacturer_ferrari', 'manufacturer_fiat', 'manufacturer_ford',
       'manufacturer_gmc', 'manufacturer_harley-davidson',
       'manufacturer_honda', 'manufacturer_hyundai', 'manufacturer_infiniti',
       'manufacturer_jaguar', 'manufacturer_jeep', 'manufacturer_kia',
       'manufacturer_land rover', 'manufacturer_lexus', 'manufacturer_lincoln',
       'manufacturer_mazda', 'manufacturer_mercedes-benz',
       'manufacturer_mercury', 'manufacturer_mini', 'manufacturer_mitsubishi',
       'manufacturer_morgan', 'manufacturer_nissan', 'manufacturer_pontiac',
       'manufacturer_porsche', 'manufacturer_ram', 'manufacturer_rover',
       'manufacturer_saturn', 'manufacturer_subaru', 'manufacturer_tesla',
       'manufacturer_toyota', 'manufacturer_volkswagen', 'manufacturer_volvo'
]


fuel_types = ['fuel_diesel', 'fuel_electric', 'fuel_gas', 'fuel_hybrid', 'fuel_other']
title_status = [
        'title_status_clean', 'title_status_lien', 'title_status_missing',
       'title_status_parts only', 'title_status_rebuilt',
       'title_status_salvage'
]
transmisions = ['transmission_automatic', 'transmission_manual',
       'transmission_other']
car_type = ['type_SUV', 'type_bus', 'type_convertible',
       'type_coupe', 'type_hatchback', 'type_mini-van', 'type_offroad',
       'type_other', 'type_pickup', 'type_sedan', 'type_truck', 'type_van',
       'type_wagon']
contidions = ['condition_excellent', 'condition_fair', 'condition_good',
       'condition_like new', 'condition_new', 'condition_salvage']


### UI
st.title("Used Car Price Predictor by Mantheous")
year = st.number_input(label="Year", format="%0i", value=2026)
milage = st.number_input(label="Milage", format="%0i", value = 100)
manufacturer = st.selectbox(label="Manufacturer", options=manufactuers)
cylinders = st.number_input(label="cylinders", format="%0i", value=4)
fuel_type = st.selectbox(label = "Fuel Type", options=fuel_types)
title = st.selectbox(label = "Title", options=title_status)
transmision = st.selectbox(label = "Transmision", options=transmisions)
c_type = st.selectbox(label = "Car Type", options=car_type)
condition = st.selectbox(label = "Condition", options=contidions)

### Response
scaler, model = load_artifacts()

if st.button("Predict Price"):
    
    X = preprocess(
          year=year, 
          milage=milage, 
          manu=manufacturer, 
          cyl=cylinders, 
          fuel=fuel_type, 
          tit=title, 
          trans=transmision, 
          c_type=c_type, 
          condition=condition, 
          scaler=scaler
    )

    prediction = model.predict(X)
    
    # Extracting the first item from the array and formatting it as currency looks a bit cleaner!
    predicted_price = np.e ** prediction[0]
    st.header(f"Price: ${predicted_price:,.2f}")

### Footer
st.text("""\
            This model explains 67% of the variation in the data. Perhaps a more
            intuitive way to view the accuracy is the average error. I probably am
            Not going to get very tangible measurements of accuracy for you. If you
            care the r^2 is like 0.45 on a logarithmic transform of the price. So uh it's
            more accurate for smaller numbers and less accurate for bigger numbers. 
            Like floating point numbers! Well, it's pretty good okay.
""")