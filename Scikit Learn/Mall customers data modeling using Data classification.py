import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
url = "https://github.com/sayooj17/task-1-data-cleaning/blob/main/mall_customers_cleaned.csv?raw=true"
df = pd.read_csv(url)

# Creating a target column based on spending score
def spend_category(score):
    if score >= 85:
        return 'High'
    elif score >= 50:
        return 'Medium'
    else:
        return 'Low'
df['spend_category'] = df['spending_score_(1-100)'].apply(spend_category)
# Encode 'gender' to numeric
df['gender_encoded'] = LabelEncoder().fit_transform(df['gender'])
# Features and target
X = df[['annual_income_(k$)', 'age', 'gender_encoded']]
y = df['spend_category']
# Encode target variable
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))
