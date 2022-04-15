import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from prettytable import PrettyTable

iris_dataset = load_iris()
iris_data = iris_dataset.data
iris_labels = iris_dataset.target

# Exercise C and D using 3 Features
print("One vs All Classification using features 1,2,4")
iris_data_f_124 = np.hstack([iris_data[:, 0:2], iris_data[:, 3].reshape(-1, 1)])
X_train_f_124, X_test_f_124, y_train_2vs13, y_test_2vs13 = train_test_split(iris_data_f_124, iris_labels, test_size=0.2, random_state=0, stratify=iris_labels)
K_folds = 4

x = PrettyTable()
x.field_names = ["SVM", "Kernel Type", "C", "Validation Accuracy", "Test Accuracy"]
print(25*" ", "USING LINEAR SVM")
for c in [0.5, 1, 10, 100, 1000]:
    model = svm.SVC(kernel='linear', C=c, decision_function_shape="ovr").fit(X_train_f_124, y_train_2vs13)
    scores = cross_val_score(model, X_train_f_124, y_train_2vs13, cv=K_folds)
    predict = model.predict(X_test_f_124)
    x.add_row(["Linear SVM", " Linear", str(c), str(scores.mean())[:5] + " ± " + str(scores.std())[0:5], str(round(accuracy_score(y_test_2vs13, predict), 5))])
print(x)


x = PrettyTable()
x.field_names = ["SVM", "Kernel Type", "C", "Degree", "Validation Accuracy", "Test Accuracy"]
print(30*" ", "USING NON LINEAR SVM")
for c in [0.5, 1, 10, 100, 1000]:
    model = svm.SVC(kernel='sigmoid', C=c, decision_function_shape="ovr").fit(X_train_f_124, y_train_2vs13)
    scores = cross_val_score(model, X_train_f_124, y_train_2vs13, cv=K_folds)
    predict = model.predict(X_test_f_124)
    x.add_row(["Non LINEAR SVM", " Sigmoid", str(c), "-", str(scores.mean())[:5] + " ± " + str(scores.std())[0:5], str(round(accuracy_score(y_test_2vs13, predict), 5))])

for c in [0.5, 1, 10, 100, 1000]:
    model = svm.SVC(kernel='rbf', C=c,decision_function_shape="ovr").fit(X_train_f_124, y_train_2vs13)
    scores = cross_val_score(model, X_train_f_124, y_train_2vs13, cv=K_folds)
    predict = model.predict(X_test_f_124)
    x.add_row(["Non LINEAR SVM", " RBF", str(c), "-", str(scores.mean())[:5] + " ± " + str(scores.std())[0:5], str(round(accuracy_score(y_test_2vs13, predict), 5))])

for d in [2, 4, 6]:
    for c in [0.5, 1, 10, 100, 1000]:
        model = svm.SVC(kernel='poly', C=c, decision_function_shape="ovr").fit(X_train_f_124, y_train_2vs13)
        scores = cross_val_score(model, X_train_f_124, y_train_2vs13, cv=K_folds)
        predict = model.predict(X_test_f_124)
        x.add_row(["Non LINEAR SVM", " Polynomial", str(c), str(d), str(scores.mean())[:5] + " ± " + str(scores.std())[0:5], str(round(accuracy_score(y_test_2vs13, predict), 5))])
print(x)