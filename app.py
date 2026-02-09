import streamlit as st
import joblib
import numpy as np

st.title("House Price Prediction App")
st.divider()
st.write("This app uses machine learning for predicting house price with given features")
st.divider()

# Load model first to get feature names
try:
    model = joblib.load("model.pkl")
    
    # Get the actual feature names the model expects
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
    elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_names_in_'):
        expected_features = model.best_estimator_.feature_names_in_
    else:
        st.error("Cannot determine expected features from model")
        st.stop()
    
    # Extract location columns from expected features
    location_cols = [f for f in expected_features if f.startswith('location_')]
    
    # User inputs
    rooms = st.number_input("Number of rooms", min_value=0, value=2)
    square = st.number_input("Square meters", min_value=0.0, value=80.0)
    floor = st.text_input("Floor (e.g., 5/9)", "5/9")
    new_building = st.selectbox("New building?", ["No", "Yes"])
    has_repair = st.selectbox("Has repair?", ["No", "Yes"])
    
    # Location selectbox
    location_names = [col.replace("location_", "") for col in location_cols]
    location = st.selectbox("Location", location_names)
    
    # Convert Yes/No to 0/1
    new_building_val = 1 if new_building == "Yes" else 0
    has_repair_val = 1 if has_repair == "Yes" else 0
    
    # Convert floor to float
    def convert_floor(x):
        if "/" in str(x):
            try:
                a, b = str(x).split("/")
                return float(a) / float(b)
            except:
                return 0.0
        try:
            return float(x)
        except:
            return 0.0
    
    floor_val = convert_floor(floor)
    
    # Predict button
    if st.button("Predict!"):
        st.balloons()
        
        # Create feature dictionary with all expected features
        feature_dict = {}
        
        # Add base features (check exact names in expected_features)
        for feat in expected_features:
            if feat == 'rooms' or feat == 'Rooms' or feat == 'number_of_rooms':
                feature_dict[feat] = rooms
            elif feat == 'square' or feat == 'Square' or feat == 'square_meters':
                feature_dict[feat] = square
            elif feat == 'floor' or feat == 'Floor':
                feature_dict[feat] = floor_val
            elif feat == 'new_building' or feat == 'New_building' or feat == 'is_new':
                feature_dict[feat] = new_building_val
            elif feat == 'has_repair' or feat == 'repair' or feat == 'Repair':
                feature_dict[feat] = has_repair_val
            elif feat.startswith('location_'):
                # Set 1 for selected location, 0 for others
                feature_dict[feat] = 1 if feat == f"location_{location}" else 0
            else:
                # For any other feature, set default value
                feature_dict[feat] = 0
        
        # Create feature array in the exact order expected by model
        features = np.array([[feature_dict[feat] for feat in expected_features]])
        
        # Predict
        prediction = model.predict(features)
        st.success(f"Predicted House Price: {prediction[0]:,.2f} AZN")
    else:
        st.write("Please use the predict button after entering values")

except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please make sure it's in the correct directory.")
except Exception as e:
    st.error(f"Error loading model or making prediction: {str(e)}")