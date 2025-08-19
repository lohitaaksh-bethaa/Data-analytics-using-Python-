#Handling the missing values by using simple imputer command
#Encoding categories using OneHotEncoder LabelEncoder
#Scaling the data using StandardScaler and MinMaxScaler
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits, make_classification
from sklearn.metrics import accuracy_score
url = "https://github.com/raorao/datasciencedojo/blob/master/Datasets/titanic.csv?raw=true"
df = pd.read_csv(url)
#Features and targets
X = df[['Pclass','Sex','Age','Fare']]
y = df['Survived']
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Pipeline for preprocessing
numeric_features = ['Age', 'Fare']
categorical_features = ['Pclass', 'Sex']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
#Combine with model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])
#train and evaluate the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))