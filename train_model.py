import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Step 1: Load the dataset
dataset = pd.read_csv('Dataset.csv')

# Encode categorical columns
categorical_columns = ['Protcol', 'Flag', 'Family', 'SeddAddress', 'ExpAddress', 'IPaddress', 'Threats']
label_encoders = {}

for col in categorical_columns:
    encoder = LabelEncoder()
    dataset[col] = encoder.fit_transform(dataset[col])
    label_encoders[col] = encoder

# Separate features (X) and target (y)
X = dataset.drop(columns=['Prediction'])
y = dataset['Prediction']

# Encode target labels
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Random Forest model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Evaluate the model
y_pred = random_forest_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Save the model and encoders
joblib.dump(random_forest_model, 'random_forest_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')
print("Model and encoders saved successfully.")
