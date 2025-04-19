import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from joblib import dump

import sklearn
print(sklearn.__version__)

# Reading a CSV (Comma-Separated Values) file into a Pandas DataFrame.
data = pd.read_csv("Breast Cancer METABRIC.csv") 

## Data preprocessing
# Impute missing values for categorical values
def impute_categorical_most_frequent(df, categorical_columns):
    imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
    return df

categorical_cols = ['Overall Survival Status','PR Status','ER Status','Hormone Therapy']
data = impute_categorical_most_frequent(data, categorical_cols)

## Removing outliers from columns 'Age at Diagnosis','Tumor Stage','Tumor Size'
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

numeric_cols = ['Age at Diagnosis','Tumor Stage','Tumor Size']
data = remove_outliers_iqr(data, numeric_cols)

# Impute missing values for numerical values
def impute_numerical_mean(df, numerical_columns):
    imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
    return df

numerical_cols = ['Age at Diagnosis','Tumor Stage','Tumor Size'] 
data = impute_numerical_mean(data, numerical_cols)

## Convert 'Overall Survival Status' to binary labels
le = LabelEncoder()
data["Overall Survival Status"] =le.fit_transform(data["Overall Survival Status"])

# Use Label Encoding for 'PR Status','ER Status' & 'Hormone Therapy'
data['PR Status'] = le.fit_transform(data['PR Status'])
data['ER Status'] = le.fit_transform(data['ER Status'])
data['Hormone Therapy'] = le.fit_transform(data['Hormone Therapy'])

## Selecting the  features
selected_features = ['PR Status','ER Status','Hormone Therapy','Age at Diagnosis','Tumor Stage','Tumor Size']
X = data[selected_features]
y = data['Overall Survival Status']

# Train the Random Forest model
model = RandomForestClassifier(bootstrap=True, criterion='entropy', min_samples_leaf=2, n_estimators=75,min_samples_split=2)
model.fit(X, y)

# Save the trained model to a file
dump(model, 'random_forest_model.joblib')


