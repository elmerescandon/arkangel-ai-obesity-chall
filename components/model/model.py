import streamlit as st
import pandas as pd
from utils.utils import process_data

def model_try(model):
    st.title("Health Form")
    st.write("This form will help you to determine your body mass index and predict your weight status.")   
    with st.form(key='health_form'):
        gender = st.selectbox("What is your gender?", ["Female", "Male"])
        age = st.number_input("What is your age?", min_value=0, max_value=120, step=1)
        height = st.number_input("What is your height (in meters)?", min_value=0.0, max_value=3.0, step=0.01)
        weight = st.number_input("What is your weight (in kilograms)?", min_value=0, max_value=300, step=1)
        
        family_overweight = st.radio("Has a family member suffered or suffers from overweight?", ["Yes", "No"])
        high_caloric_food = st.radio("Do you eat high caloric food frequently?", ["Yes", "No"])
        vegetables = st.radio("Do you usually eat vegetables in your meals?", ["Never", "Sometimes", "Always"])
        
        main_meals = st.radio("How many main meals do you have daily?", ["Between 1 and 2", "Three", "More than three"])
        food_between_meals = st.radio("Do you eat any food between meals?", ["No", "Sometimes", "Frequently", "Always"])
        
        smoking = st.radio("Do you smoke?", ["Yes", "No"])
        water_intake = st.radio("How much water do you drink daily?", ["Less than a liter", "Between 1 and 2 L", "More than 2 L"])
        calorie_monitoring = st.radio("Do you monitor the calories you eat daily?", ["Yes", "No"])
        
        physical_activity = st.radio("How often do you have physical activity?", ["I do not have", "1 or 2 days", "2 or 4 days", "4 or 5 days"])
        tech_usage = st.radio("How much time do you use technological devices such as cell phone, videogames, television, computer and others?", ["0-2 hours", "3-5 hours", "More than 5 hours"])
        
        alcohol_consumption = st.radio("How often do you drink alcohol?", ["I do not drink", "Sometimes", "Frequently", "Always"])
        transportation = st.radio("Which transportation do you usually use?", ["Automobile", "Motorbike", "Bike", "Public Transportation", "Walking"])
        
        submit_button = st.form_submit_button(label='Enviar formulario')

    if submit_button:
        if gender and age and height and weight and family_overweight and high_caloric_food and vegetables and main_meals and food_between_meals and smoking and water_intake and calorie_monitoring and physical_activity and tech_usage and alcohol_consumption and transportation:
            data = {
                'Gender': [gender], # Done
                'Age': [age], # Done
                'Height': [height], # Done
                'Weight': [weight], # Done
                'family_history_with_overweight': [family_overweight.lower()], # Done
                'FAVC': [high_caloric_food.lower()], # Done
                'FCVC': [1 if vegetables == "Never" else (2 if vegetables == "Sometimes" else 3)], # Done
                'NCP': [1 if main_meals == "Between 1 and 2" else (3 if main_meals == "Three" else 4)], # Done
                'CAEC': [food_between_meals.lower() if food_between_meals=="No" else food_between_meals  ], # Done
                'SMOKE': [smoking.lower()], #Done
                'CH2O': [1 if water_intake == "Less than a liter" else (2 if water_intake == "Between 1 and 2 L" else 3)], # Done
                'SCC': [calorie_monitoring.lower()], # Done
                'FAF': [0 if physical_activity == "I do not have" else (1 if physical_activity == "1 or 2 days" else (2 if physical_activity == "2 or 4 days" else 3))], # Done
                'TUE': [0 if tech_usage == "0-2 hours" else (1 if tech_usage == "3-5 hours" else 2)], # Done
                'CALC': [alcohol_consumption.lower() if alcohol_consumption == "No" else alcohol_consumption], # Done],
                'MTRANS': ["Public_Transportation" if transportation == "Public Transportation" else transportation] # Done
            }

            df = pd.DataFrame(data)
            result = process_data(df, model)
            st.info(f"Your body mass index is: {result}")
        else:
            st.write("Please fill in all the fields before submitting the form.")
