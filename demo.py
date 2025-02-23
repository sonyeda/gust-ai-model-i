import streamlit as st
from datetime import date
import pandas as pd
import random
import joblib
import xgboost
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient
import os

print(os.getcwd())

# MongoDB Connection
client = MongoClient("mongodb+srv://sonyeda601:buodR9tHY0aIjd4A@cluster0.gbw0x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

db = client["hotel_guests"]
new_bookings_collection = db["new_bookings"]

# Helper function to load Excel files
def load_excel_file(filename, rename_cols=None):
    try:
        if os.path.exists(filename):
            df = pd.read_excel(filename, engine='openpyxl')
            if rename_cols:
                df.rename(columns=rename_cols, inplace=True)
            st.write(f"✅ Loaded {filename} successfully!")
            st.write(f"Columns: {df.columns.tolist()}")
            return df
        else:
            st.error(f"❌ File not found: {filename}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading {filename}: {str(e)}")
        return pd.DataFrame()

st.title("🏨 Hotel Booking Form")

# Customer ID handling
has_customer_id = st.radio("Do you have a Customer ID?", ("Yes", "No"))

if has_customer_id == "Yes":
    customer_id = st.text_input("Enter your Customer ID", "")
else:
    customer_id = random.randint(10001, 99999)
    st.write(f"Your generated Customer ID: {customer_id}")

# Booking form fields
name = st.text_input("Enter your name", "")
checkin_date = st.date_input("Check-in Date", min_value=date.today())
checkout_date = st.date_input("Check-out Date", min_value=checkin_date)
age = st.number_input("Enter your age", min_value=18, max_value=120, step=1)
stayers = st.number_input("How many stayers in total?", min_value=1, max_value=3, step=1)
cuisine_options = ["South Indian", "North Indian", "Multi"]
preferred_cuisine = st.selectbox("Preferred Cuisine", cuisine_options)
booking_options = ["Yes", "No"]
preferred_booking = st.selectbox("Do you want to book through points?", booking_options)

special_requests = st.text_area("Any Special Requests? (Optional)", "")

# Handle booking submission
if st.button("Submit Booking"):
    if name and customer_id:
        new_data = {
            'customer_id': int(customer_id),
            'Preferred Cuisine': preferred_cuisine,
            'age': age,
            'check_in_date': checkin_date,
            'check_out_date': checkout_date,
            'booked_through_points': 1 if preferred_booking == 'Yes' else 0,
            'number_of_stayers': stayers
        }

        new_df = pd.DataFrame([new_data])
        new_df['check_in_date'] = pd.to_datetime(new_df['check_in_date'])
        new_df['check_out_date'] = pd.to_datetime(new_df['check_out_date'])
        new_df['stay_duration'] = (new_df['check_out_date'] - new_df['check_in_date']).dt.days

        # Insert booking into MongoDB
        new_bookings_collection.insert_one(new_df.iloc[0].to_dict())

        # Load customer and cuisine data
        customer_features = load_excel_file('customer_features.xlsx')
        customer_dish = load_excel_file('customer_dish.xlsx')
        cuisine_features = load_excel_file('cuisine_features.xlsx', rename_cols={'Preferred Cusine': 'Preferred Cuisine'})
        cuisine_dish = load_excel_file('cuisine_dish.xlsx', rename_cols={'Preferred Cusine': 'Preferred Cuisine'})

        # Validate loaded data
        for df, name, col in [(customer_features, 'customer_features', 'customer_id'), 
                              (cuisine_features, 'cuisine_features', 'Preferred Cuisine'),
                              (customer_dish, 'customer_dish', 'customer_id'),
                              (cuisine_dish, 'cuisine_dish', 'Preferred Cuisine')]:
            if df.empty or col not in df.columns:
                st.error(f"❌ Missing '{col}' column in {name} file or the file is empty!")
                st.stop()

        # Merge data
        new_df = new_df.merge(customer_features, on='customer_id', how='left')
        new_df = new_df.merge(cuisine_features, on='Preferred Cuisine', how='left')
        new_df = new_df.merge(customer_dish, on='customer_id', how='left')
        new_df = new_df.merge(cuisine_dish, on='Preferred Cuisine', how='left')

        new_df.drop(['customer_id', 'check_in_date', 'check_out_date'], axis=1, inplace=True)

        # Load model and encoders with error handling
        try:
            if not os.path.exists('xgb_model_dining.pkl') or not os.path.exists('encoder.pkl') or not os.path.exists('label_encoder.pkl'):
                st.error("❌ Model or encoder files not found! Please upload 'xgb_model_dining.pkl', 'encoder.pkl', and 'label_encoder.pkl'.")
                st.stop()

            model = joblib.load('xgb_model_dining.pkl')
            encoder = joblib.load('encoder.pkl')
            label_encoder = joblib.load('label_encoder.pkl')

            st.success("✅ Model and encoders loaded successfully!")

        except Exception as e:
            st.error(f"❌ Error loading model or encoders: {e}")
            st.stop()

        # Handle missing columns carefully
        required_columns = ['Preferred Cuisine', 'most_frequent_dish', 'cuisine_popular_dish']
        missing_columns = [col for col in required_columns if col not in new_df.columns]

        if missing_columns:
            st.error(f"❌ Missing required columns in merged data: {', '.join(missing_columns)}")
            st.stop()

        # Handle encoding
        try:
            categorical_cols = new_df.select_dtypes(include=['object']).columns.tolist()
            encoded_test = encoder.transform(new_df[categorical_cols])
            encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))
            new_df = pd.concat([new_df.drop(columns=categorical_cols), encoded_test_df], axis=1)

        except Exception as e:
            st.error(f"❌ Error during data encoding: {e}")
            st.stop()

        # Check feature alignment
        feature_list = load_excel_file('features.xlsx').iloc[:, 0].tolist()
        missing_features = [feat for feat in feature_list if feat not in new_df.columns]

        if missing_features:
            st.warning(f"⚠ Missing features in data: {', '.join(missing_features)}")
            st.stop()

        new_df = new_df[feature_list]

        # Make predictions
        y_pred_prob = model.predict_proba(new_df)
        dish_names = label_encoder.classes_
        top_3_indices = np.argsort(-y_pred_prob, axis=1)[:, :3]
        top_3_dishes = dish_names[top_3_indices]

        # Show booking details and discounts
        st.success(f"✅ Booking Confirmed for {name} (Customer ID: {customer_id})!")
        st.write(f"*Check-in:* {checkin_date}")
        st.write(f"*Check-out:* {checkout_date}")
        st.write(f"*Age:* {age}")
        st.write(f"*Preferred Cuisine:* {preferred_cuisine}")

        if special_requests:
            st.write(f"*Special Requests:* {special_requests}")

        dishes = [dish.lower() for dish in top_3_dishes[0]]
        thali_dishes = [dish for dish in dishes if "thali" in dish]
        other_dishes = [dish for dish in dishes if "thali" not in dish]

        st.write("Discounts for you!")
        if thali_dishes:
            st.write(f"Get 20% off on {', '.join(thali_dishes)}")
        if other_dishes:
            st.write(f"Get 15% off on {', '.join(other_dishes)}")

        st.write("Check your coupon code on your email!")

    else:
        st.warning("⚠ Please enter your name and Customer ID to proceed!")
