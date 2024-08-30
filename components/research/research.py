import streamlit as st
import pandas as pd

def research():
    @st.cache_data
    def load_results(file_path):
        return pd.read_csv(file_path)

    results = load_results("./results/results.csv")
    results_2 = load_results("./results/results_wo_data.csv")

    # Overview
    with st.expander("Overview"):
        st.write('This is a solution for the Arkangel AI Challenge. The goal is to predict the risk of cardiovascular disease (CVR) based on the obesity level of a patient. The dataset contains information about the obesity level of patients and their CVR risk. The dataset is divided into two files: train.csv and test.csv. The train.csv file contains the training data, while the test.csv file contains the test data. The goal is to build a machine learning model that can predict the CVR risk of patients based on their obesity level.')

    # Data Exploration
    with st.expander("Data Exploration"):
        st.write("The data included 20,758 entries with 17 features with 8 numerical and 8 categorical. The target variable is 'CVRisk'. [1]")

    # Data Preprocessing
    with st.expander("Data Preprocessing"):
        st.write('The data preprocessing steps include: ')
        st.markdown("""
                    1) Standarize numerical data.
                    2) Test with two encoding methods for categorical data. \n
                        2.1. _**Label Encoding**_: Encode categorical data as integers, most of the variables include a range of selection. \n
                        2.2. _**One-Hot Encoding**_: Create a binary column for each category, variables with collinearity are reduced. \n
                    3) Feature scaling.
                    4) Splitting the data into training and testing sets.
                    """)

    with st.expander("Feature Importance"):
        st.write('According to [4], the dataset include more relevant information for the prediction of the CVR risk, such as:')
        st.markdown(""" 
                    1) Physical Activity
                    2) Alcohol consumption
                    3) Use of technology devices
                    4) Frequency of high-caloric food consumption
                    5) Frequency of vegetables consumption"""); 
        st.write('An evaluation considering the chosen classification models was performed to determine the feature importance.')

    # Model Selection
    with st.expander("Model Selection"):
        st.write('_**Decision Tree [2]**_')
        st.write('1) Dataset contain both numerical and categorical variables.')
        st.write('2) Easy to interpret and visualize, so a feature importance can be extracted.')
        st.write('3) Features present a non-linear relationship with the target variable.')
        st.write('4) Features have low correlation with each other.')

        st.write('_**Random Forest**_')
        st.write('1) Dataset contain both numerical and categorical variables.')
        st.write('2) Random Forest is an ensemble method that combines multiple decision trees.')
        st.write('3) Random Forest is robust to overfitting and can handle high-dimensional data.')

        st.write('_**Gradient Boosting**_')
        st.write('1) Dataset contain both numerical and categorical variables.')
        st.write('2) Gradient Boosting is an ensemble method that builds trees sequentially.')
        st.write('3) Gradient Boosting is robust to overfitting and can handle high-dimensional data.')

        st.info('_Other vanilla models were tested but did not perform as well as the ensemble methods. See "data_models.ipynb" for more details._')

    # Results
    with st.expander("Results"):
        st.write('The results of the model evaluation are shown in the table below:')
        st.write(results)
        st.write(results_2)

    # References
    with st.expander("References"):
        st.write('[1] F. M. Palechor and A. De la Hoz Manotas, "Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico," Data in Brief, vol. 25, p. 104344, 2019.')
        st.write('[2] De-La-Hoz-Correa, E., Mendoza Palechor, F., De-La-Hoz-Manotas, A., Morales Ortega, R., and Sánchez Hernández, A. B., "Obesity level estimation software based on decision trees," Universidad de la Costa, 2019.')
        st.write('[3] F. H. Yagin, M. Gülü, Y. Gormez, A. Castañeda-Babarro, C. Colak, G. Greco, F. Fischetti, and S. Cataldi, "Estimation of Obesity Levels with a Trained Neural Network Approach optimized by the Bayesian Technique," Applied Sciences, vol. 13, no. 6, Art. no. 3875, 2023. [Online].')
        st.write('[4] W. Reade and A. Chow, "Multi-Class Prediction of Obesity Risk," Kaggle, 2024. [Online]. Available: https://kaggle.com/competitions/playground-series-s4e2')
