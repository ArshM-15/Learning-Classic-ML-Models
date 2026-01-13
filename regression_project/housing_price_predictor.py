# House Price Predictor - User Input

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value < 0:
                print("Error: Value must be positive. Try again.")
                continue
            return value
        except ValueError:
            print("Error: Invalid number. Try again.")


def get_int_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value < 0:
                print("Error: Value must be positive. Try again.")
                continue
            return value
        except ValueError:
            print("Error: Invalid number. Try again.")


def get_yes_no_input(prompt):
    while True:
        value = input(prompt).lower().strip()
        if value in ['yes', 'no']:
            return value
        print("Error: Enter 'yes' or 'no'. Try again.")


def get_furnishing_input(prompt):
    while True:
        value = input(prompt).lower().strip()
        if value in ['furnished', 'semi-furnished', 'unfurnished']:
            return value
        print("Error: Enter 'furnished', 'semi-furnished', or 'unfurnished'. Try again.")


print("="*60)
print("HOUSE PRICE PREDICTOR")
print("="*60)

# Load and train model
print("\nLoading model...")
dataset = pd.read_csv('housing.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

le = LabelEncoder()
for i in [4, 5, 6, 7, 8, 10]:
    X[:, i] = le.fit_transform(X[:, i])

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [11])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")
print("\n" + "="*60)
print("Enter house details:")
print("="*60)

# Get user input with validation
area = get_float_input("\nArea (sq ft): ")
bedrooms = get_int_input("Bedrooms: ")
bathrooms = get_int_input("Bathrooms: ")
stories = get_int_input("Stories: ")
mainroad = get_yes_no_input("Main road (yes/no): ")
guestroom = get_yes_no_input("Guest room (yes/no): ")
basement = get_yes_no_input("Basement (yes/no): ")
hotwaterheating = get_yes_no_input("Hot water heating (yes/no): ")
airconditioning = get_yes_no_input("Air conditioning (yes/no): ")
parking = get_int_input("Parking spaces: ")
prefarea = get_yes_no_input("Preferred area (yes/no): ")
furnishingstatus = get_furnishing_input(
    "Furnishing status (furnished/semi-furnished/unfurnished): ")

# Encode inputs
mainroad = 1 if mainroad == 'yes' else 0
guestroom = 1 if guestroom == 'yes' else 0
basement = 1 if basement == 'yes' else 0
hotwaterheating = 1 if hotwaterheating == 'yes' else 0
airconditioning = 1 if airconditioning == 'yes' else 0
prefarea = 1 if prefarea == 'yes' else 0

if furnishingstatus == 'furnished':
    furnish = [1, 0, 0]
elif furnishingstatus == 'semi-furnished':
    furnish = [0, 1, 0]
else:
    furnish = [0, 0, 1]

# Create input array
user_input = furnish + [area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                        hotwaterheating, airconditioning, parking, prefarea]
user_input = np.array(user_input).reshape(1, -1)

# Predict
prediction = model.predict(user_input)[0]

print("\n" + "="*60)
print(f"PREDICTED PRICE: ${prediction:,.2f}")
print("="*60)
