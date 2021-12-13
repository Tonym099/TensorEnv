import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

# Linear regression uses correlated data to make a prediction

data = pd.read_csv("/Users/tony/PythonProjects/TensorEnv/Data/student/student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"  # "predict" is the label, the value(s) we're trying to get

x = np.array(data.drop([predict], 1))  # return a array of training data without the predict variable
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)  # Need two copies
# of this in order for the loaded in model to be tested

"""
best = 0
while best < 0.95:
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    # split the x and y array into four total arrays, with the "train" array used to train the ai, and "test" array
    # to test the ai with information that it has not seen thus memorized before


    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best:
        best = acc
        print(best)
        with open("studentmodel.pickle", "wb") as f:  # pickle saves the trained AI instead of retraining it every time
            pickle.dump(linear, f)  # Can comment out the training portion and the program will work
"""

pickle_in = open("studentmodel.pickle", "rb")  # opens the saved model
linear = pickle.load(pickle_in)  # loads the models


predictions = linear.predict(x_test)

"""
print("Coefficients: \n", linear.coef_)
print("Intercepts: \n", + linear.intercept_)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
"""

p = 'studytime'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])  # Plots the points with x=p and y=G3
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()