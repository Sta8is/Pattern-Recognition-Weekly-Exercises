from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay


iris_dataset = load_iris()
iris_data = iris_dataset.data
iris_labels = iris_dataset.target
target_names = list(iris_dataset.target_names)
dict_map = {i: target_names[i] for i in range(len(target_names))}
# Train - Test Split
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_labels, test_size=0.2, random_state=1, stratify=iris_labels)
K_folds = 4


model = Perceptron(tol=1e-4, max_iter=1000, n_jobs=1)
model.fit(X_train, y_train)
predict_train = model.predict(X_train)
print("Train accuracy:", accuracy_score(predict_train, y_train))
scores = cross_val_score(model, X_train, y_train, cv=K_folds)
print("Validation Accuracy ", ": %0.5f ± %0.5f" % (scores.mean(), scores.std()))
predict = model.predict(X_test)
clf_report = classification_report(y_test,predict, target_names=target_names, digits=5)
print("Classification Report on Test set:\n", clf_report)
y_test_names = [dict_map[i] for i in y_test]
predict_names = [dict_map[i] for i in predict]

weights = model.coef_
biases = model.intercept_
print("Model Weights: \n", weights)
print("Model Biases: ", biases)

ConfusionMatrixDisplay.from_predictions(y_test_names, predict_names, labels=target_names)
plt.show()