import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
dataset = pd.read_csv('D:/OneDrive/Desktop/Desktop/aiml_mini/aiml_mini/Dataset (1).csv')  # Replace with your dataset filename

# Check and clean the dataset
print("Dataset Overview:")
print(dataset.info())
print("Unique values in 'Prediction':", dataset['Prediction'].unique())

# Preprocess the dataset
label_encoders = {}
for col in dataset.select_dtypes(include='object').columns:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    label_encoders[col] = le

# Save LabelEncoder for the target column separately
if 'Prediction' in label_encoders:
    le_prediction = label_encoders.pop('Prediction')  # Remove from dictionary and save separately
    joblib.dump(le_prediction, 'prediction_label_encoder.pkl')
else:
    raise ValueError("The 'Prediction' column is missing or was not encoded correctly.")

# Standardize numerical features
scaler = StandardScaler()
numerical_columns = dataset.select_dtypes(include='number').columns.drop('Prediction')
dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

# Split data into features (X) and target (y)
X = dataset.drop('Prediction', axis=1)
y = dataset['Prediction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Save the model and preprocessing objects
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and preprocessing artifacts saved successfully!")
