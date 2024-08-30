import pandas as pd
from sklearn.preprocessing import LabelEncoder

resultToLabel = {
    'Obesity_Type_I': "Obesity Type I",
    'Obesity_Type_II': "Obesity Type II",
    'Obesity_Type_III': "Obesity Type III",
    'Overweight_Level_I': "Overweight Level I",
    'Overweight_Level_II': "Overweight Level II",
    'Normal_Weight': "Normal Weight",
    'Insufficient_Weight': "Insufficient Weight"
}

def analyzeDataTypes(df):
    continuous_vars = []
    categorical_vars = []
    for column in df.columns:
        if df[column].dtype == 'object':
            categorical_vars.append(column)
        else:
            continuous_vars.append(column)
    return continuous_vars, categorical_vars


def preprocessingOneHotEncoding(df, variables):
    if 'NObeyesdad' in variables:
        variables.remove('NObeyesdad')
    df_train = pd.get_dummies(df, columns=variables)
    # Eliminate multicollinearity
    df_train.drop('Gender_Female', axis=1, inplace=True)
    df_train.drop('family_history_with_overweight_yes', axis=1, inplace=True)
    df_train.drop('FAVC_no', axis=1, inplace=True)
    df_train.drop('SMOKE_yes', axis=1, inplace=True)
    df_train.drop('SCC_yes', axis=1, inplace=True)
    return df_train

def preprocessingWithLabelEncoder(df, variables):
    le = LabelEncoder()
    for column in variables:
        df[column] = le.fit_transform(df[column])
    return df

def prepareData(df):
    df = df.dropna()
    X = df.drop_duplicates()
    return X

def process_data(df, model):
    continuous_vars, categorical_vars = analyzeDataTypes(df)
    df = preprocessingWithLabelEncoder(df, categorical_vars)
    result = model.predict(df)
    return resultToLabel[result[0]]