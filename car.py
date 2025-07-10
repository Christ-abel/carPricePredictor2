#load data from csv file and store it in mysql database

import pandas as pd
from sqlalchemy import create_engine
df =pd.read_csv(r'C:\Users\HP\OneDrive\Documents\synthetic_car_data.csv')
print(df.isnull().sum()) #Check for null values

# Create a connection to the MySQL database
engine =create_engine("mysql+mysqlconnector://Bellah:walumbe1@localhost/carprice2")
df.to_sql(name='records',con=engine,if_exists='replace',index=False)
print("Data inserted successfully into MySQL database")

# Separate the independent and dependent variables
X = df.drop(columns=['Price'])
y = df['Price']

#separate the categorical and numerical columns

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

#print the categorical and numerical columns

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# Print the first few rows of the DataFrame to verify the data
print(df.head())

#Encode categorical variables using one-hot encoding
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Scale numerical variables using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

#combine the encoded categorical and scaled numerical variables
import numpy as np
X_final = np.hstack((X_encoded[numerical_cols].values, X_encoded.drop(columns=numerical_cols).values))

# standardize the independent variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_final = scaler.fit_transform(X_final)

#Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

#Train a linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)  
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Save the model using joblib
import joblib
joblib.dump(model, 'car_price_model.pkl')
joblib.dump(scaler, 'car_price_scaler.pkl')
model_columns = X_encoded.columns.tolist()
joblib.dump(model_columns, 'car_price_model_columns.pkl')
print("Model saved successfully as 'car_price_model.pkl'")



