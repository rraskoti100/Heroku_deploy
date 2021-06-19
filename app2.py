import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("IRIS.csv")

#input and output variable selection
X = df.iloc[:, :-1]
y = df.iloc[:,-1]

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


#Decision tree model building
tree = DecisionTreeClassifier()
model = tree.fit(X_train, y_train)
pred = model.predict(X_test)

#printing accuracy of the model
print("Accuracy of model is: ", accuracy_score(y_test, pred))

#saving model to a file using pickle
pickle.dump(model, open('model_iris.pkl', 'wb'))

#load the model
model = pickle.load(open('model_iris.pkl', 'rb'))
print(model.predict([[5.1, 3.5, 1.4, 0.2]]))