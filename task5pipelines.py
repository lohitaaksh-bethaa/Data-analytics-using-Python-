from sklearn.datasets import load_iris, load_digits, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import pandas as pd 
import matplotlib.pyplot as plt
import joblib 

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)

print("🔹 Dataset Shape:", df.shape)
print("df.head()")

# Features and targets
X = df.drop('Outcome', axis=1)
y = df['Outcome']

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creating the pipeline
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
# Train the model
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, 'diabetes_model.pkl')

# Predict and evaluate
y_pred = pipeline.predict(X_test)

#Evaluation metrics
print("/n Model Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

#cross-validation evaluation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print("/n Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix Diabetes Prediction')
plt.show()
loaded_pipeline = joblib.load('diabetes_model.pkl')
#If entering a new data into the dataset for prediction
new_patient = [[2, 120, 70, 35, 0, 30.5, 0.5, 25]]  # Example patient data
prediction = loaded_pipeline.predict(new_patient)[0]
print("/n New Patient Prediction:", "Diabetic" if prediction == 1 else "Non-Diabetic")

