import pandas as pd  # For handling datasets
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For better visualizations
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.tree import DecisionTreeClassifier  # Decision Tree model
from sklearn.ensemble import RandomForestClassifier  # Random Forest model

#  Load dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

#   Drop irrelevant columns (these don't contribute to stroke prediction)
df.drop(columns=['ever_married', 'work_type', 'Residence_type', 'id'], inplace=True, errors='ignore')

# Handle missing values (Remove rows where 'bmi' is NaN)
df.dropna(subset=['bmi'], inplace=True)

# Convert categorical values to numeric
df = df[df['gender'] != 'Other']  # Remove 'Other' category from 'gender' if present
df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0})  # Convert 'Male' to 1, 'Female' to 0

# One-hot encode 'smoking_status' (Convert categorical column into multiple binary columns)
df = pd.get_dummies(df, columns=['smoking_status'], drop_first=True)  
# Example: If smoking_status has ['never smoked', 'smokes', 'formerly smoked'], it creates:
# 'smoking_status_formerly smoked' → 1 if formerly smoked, 0 otherwise
# 'smoking_status_smokes' → 1 if smokes, 0 otherwise

# Define features (X) and target variable (y)
X = df.drop(columns=['stroke'])  # All columns except 'stroke' are features
y = df['stroke']  # Target variable: 0 (No stroke) or 1 (Stroke)

# Train-Test Split (80% training data, 20% test data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.80, random_state=42, stratify=y
)  # Stratify=y ensures class distribution is maintained in both sets

# Standardize numerical features (for Logistic Regression only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)  # Transform test data using the same scaler


# 🔹 Data Visualization

## 1️⃣ Stroke Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette='coolwarm')
plt.title("Distribution of Stroke Cases")
plt.xlabel("Stroke (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

## 2️⃣ Box Plot - Age vs Stroke
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['stroke'], y=df['age'], palette='coolwarm')
plt.title("Age Distribution by Stroke Status")
plt.xlabel("Stroke (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.show()

## 3️⃣ Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

## 4️⃣ Scatter Plot - BMI vs Glucose Level (colored by Stroke)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['bmi'], y=df['avg_glucose_level'], hue=df['stroke'], palette='coolwarm')
plt.title("BMI vs Glucose Level (Colored by Stroke)")
plt.xlabel("BMI")
plt.ylabel("Avg Glucose Level")
plt.legend(title="Stroke")
plt.show()

# Define classification models
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42),  # Balanced class weights to handle imbalance
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),  # Max depth to prevent overfitting
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),  # 100 decision trees
}

accuracy_scores = {}  # Dictionary to store accuracy of each model

# Train and Evaluate each model
for model_name, model in models.items():
    print(f"\n🔹 Training {model_name}...")  # Display which model is being trained
    
    # Apply feature scaling only for Logistic Regression
    if model_name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)  # Train the model
        y_pred = model.predict(X_test_scaled)  # Predict on test set
    else:
        model.fit(X_train, y_train)  # Train without scaling
        y_pred = model.predict(X_test)  # Predict on test set

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores[model_name] = acc  # Store accuracy in dictionary

    # Display model performance
    print(f"{model_name} Accuracy: {acc:.2f}")  
    print(classification_report(y_test, y_pred, zero_division=1))  # Show precision, recall, f1-score

# Determine the Best Model (based on highest accuracy)
best_model = max(accuracy_scores, key=accuracy_scores.get)
print(f"\n Best Model: {best_model} with {accuracy_scores[best_model]:.2f} accuracy.")