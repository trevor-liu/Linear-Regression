import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

# x is the data for testing and y is the attribute that we want to predict
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# split the data into input and output for testing and training with 10% testing and 90% training
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()

# training our model and get acc using test data
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

# Getting the coefficient(slope) of the variable and the intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# predict base using input testing data
predictions = linear.predict(x_test)

# x_test is the data that we test with, y_test is the actual result
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
