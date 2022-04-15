import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from prettytable import PrettyTable

iris_dataset = load_iris()
iris_data = iris_dataset.data
iris_labels = iris_dataset.target


# Exercise A and B using 4 Features
print("Classification of class 2 with classes 1 and 3 using features 1,2,3,4")
binary_labels_2vs13 = np.where(iris_labels == 1, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(iris_data, binary_labels_2vs13, test_size=0.2, random_state=0, stratify=binary_labels_2vs13)
K_folds = 4

x = PrettyTable()
x.field_names = ["SVM", "Kernel Type", "C", "Validation Accuracy", "Test Accuracy"]
print(25*" ", "USING LINEAR SVM")
for c in [0.5, 1, 10, 100, 1000]:
    model = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=K_folds)
    predict = model.predict(X_test)
    x.add_row(["Linear SVM", " Linear", str(c), str(scores.mean())[:5]+" ± "+str(scores.std()*2)[0:5], str(round(accuracy_score(y_test, predict), 5))])
print(x)

x = PrettyTable()
x.field_names = ["SVM", "Kernel Type", "C", "Degree", "Validation Accuracy", "Test Accuracy"]
print(30*" ", "USING NON LINEAR SVM")
for c in [0.5, 1, 10, 100, 1000]:
    model = svm.SVC(kernel='sigmoid', C=c).fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=K_folds)
    predict = model.predict(X_test)
    x.add_row(["Non LINEAR SVM", " Sigmoid", str(c), "-", str(scores.mean())[:5] + " ± " + str(scores.std())[0:5], str(round(accuracy_score(y_test, predict), 5))])

for c in [0.5, 1, 10, 100, 1000]:
    model = svm.SVC(kernel='rbf', C=c).fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=K_folds)
    predict = model.predict(X_test)
    x.add_row(["Non LINEAR SVM", " RBF", str(c), "-", str(scores.mean())[:5] + " ± " + str(scores.std())[0:5], str(round(accuracy_score(y_test, predict), 5))])

for d in [2, 4, 6]:
    for c in [0.5, 1, 10, 100, 1000]:
        model = svm.SVC(kernel='poly', C=c).fit(X_train, y_train)
        scores = cross_val_score(model, X_train, y_train, cv=K_folds)
        predict = model.predict(X_test)
        x.add_row(["Non LINEAR SVM", " Polynomial", str(c), str(d), str(scores.mean())[:5] + " ± " + str(scores.std())[0:5], str(round(accuracy_score(y_test, predict), 5))])
print(x)
