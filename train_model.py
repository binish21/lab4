import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("fish.csv")  # Ensure this file exists in your directory

# Select features and target variable (Modify as per your dataset structure)
X = df.drop(columns=["Weight"])  # 'Weight' should be your target column
y = df["Weight"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open("fish_market_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")
