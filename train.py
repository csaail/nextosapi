import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the data
data = pd.read_csv("/mnt/data/Data.csv")

# Encode categorical variables
label_encoders = {}
for column in data.columns[:-2]:  
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Encode the target variable
target_encoder = LabelEncoder()
data['preferred_os'] = target_encoder.fit_transform(data['preferred_os'])

# Prepare features and target
X = data.drop(columns=['preferred_os', 'reason'])
y = data['preferred_os']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and encoders
joblib.dump(model, "os_recommendation_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
