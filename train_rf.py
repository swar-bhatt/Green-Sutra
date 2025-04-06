import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv('updated_india_agri_data.csv')

# Features and target
features = ['State', 'Soil Type', 'Previous Crop', 'Fertilizer Used', 'Water Hardness',
            'Livestock', 'Resources', 'Temperature', 'Rainfall']
target = 'Recommend Crop'

# Encode categorical variables
label_encoders = {}
for col in features + [target]:
    if df[col].dtype == 'object':
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))

# Split the data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Print accuracy
train_accuracy = rf_model.score(X_train, y_train) * 100
test_accuracy = rf_model.score(X_test, y_test) * 100
print(f"Random Forest Training Accuracy: {train_accuracy:.2f}%")
print(f"Random Forest Testing Accuracy: {test_accuracy:.2f}%")

# Save the model and label encoders
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')

print("Model and label encoders saved successfully.")