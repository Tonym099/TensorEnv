"""
K-Nearest Neighbors
The program has a data point and classifies the data point with the group that has the most similar traits

K is a hyper parameter and represent the amount of neighbors the program will look for
The program will find the K nearest neighbors of the point and will classify the data into whichever group is most
represented in the K amount of neighbors

Possible to pick too large of a value for K
Point P1 might be closest to Group G1, but when K is too large, because G2 is larger than G1, the majority of K may be
G2, resulting in P1 being incorrectly classified into G2 instead of G1

Limitation
Useless to save the AI as the program will have to check the distance from the point to every other point every time in
order to determine the K closest neighbors
"""
# Pandas read the first line of data as the attributes

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing  # preprocessing helps convert nonnumerical values to integer values

data = pd.read_csv("/Users/tony/PythonProjects/TensorEnv/Data/Car Data Set/car.data")

le = preprocessing.LabelEncoder()  # takes the labels and convert them into integer values
buying = le.fit_transform(list(data["buying"]))  # takes the entire buying colummn and turns it into a integer list
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))  # zip merges all the list into a tuple
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=7)  # parameter is K

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

"""
best_acc = 0
best_k = 0
for i in range(1, 10):
    for n in range(15):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        model = KNeighborsClassifier(n_neighbors=i)  # parameter is K

        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)

        if acc > best_acc:
            best_k = i
            best_acc = acc
            print("Best K:", best_k)
            print("Best Acc", acc , "\n")
"""

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(x_test)):
    print("Predicted:", names[predicted[x]], "     Data:", x_test[x], "    Actual:", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 7, True)
    print("N:", n, "\n")  # will return the distance and the index of the closest neighbors
