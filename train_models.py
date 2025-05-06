#Import the necessary libraries 
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Starting with loading the data
data_path = "data/Salary Data.csv"
df = pd.read_csv(data_path)

# This will remove rows where salary value is missing 
df = df.dropna(subset=["Salary"])


# Converting text data to numbers 
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
education_map = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
df["Education Level"] = df["Education Level"].map(education_map)

# So in this step choose the features and the target
features = ["Age", "Gender", "Education Level", "Years of Experience"]
target = "Salary"

X = df[features]
y = df[target]

# Splitt the data into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree model it will also calculete the performance of the model
dt_model = DecisionTreeRegressor(random_state=0)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)

# And train the random forest model it will also calculete the performance of the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# Comparing both models
print("Decision Tree:  MAE =", round(dt_mae, 2), "| R² =", round(dt_r2, 4))
print("Random Forest: MAE =", round(rf_mae, 2), "| R² =", round(rf_r2, 4))

# Save both models
os.makedirs("models", exist_ok=True)
joblib.dump(dt_model, "models/decision_tree_model.pkl")
joblib.dump(rf_model, "models/random_forest_model.pkl")
