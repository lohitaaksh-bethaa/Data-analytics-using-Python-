from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression#ML supervised learning
from sklearn.datasets import load_iris, load_digits, make_classification
iris = load_iris()
x,y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train)
model = LogisticRegression()
model.fit(x_train, y_train)
