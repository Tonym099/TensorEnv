"""
SVM attempts to create a hyperplane
Hyperplane divides the data with a linear border ie: line, plane
Exist exactly between the two closest point of each group

There are multiple possible hyperplanes
The best hyperplane is the hyperplane, that is the greatest distance away from the closest point of each group

The distance between the hyperplane and its closest point is the margin
The greater the margin, the greater the separation between the two classes and the more accurate the prediction will be

Kernels
A function that takes existing data points and bring them into a higher dimension
Used when there is no clear/good delineation/hyperplane between two classes
Typically use an existing kernel

Soft Margin
Allows some outlier points to cross over the hyperplane in order to create a hyperplane with a greater margin

Hard Margin
Only allows hyperplanes that divide all points with no exceptions
"""

import sklearn
from sklearn import datasets
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier


cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# print(x_train, y_train)
classes = ('malignant', 'benign')

clf = svm.SVC(kernel="linear", C=2)  # type of function, number of allowed exceptions
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)