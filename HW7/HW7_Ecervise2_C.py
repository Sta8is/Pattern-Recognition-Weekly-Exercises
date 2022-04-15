from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier  # Multi-Layer Perceptron Classifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from prettytable import PrettyTable

iris_dataset = load_iris()
iris_data = iris_dataset.data
iris_labels = iris_dataset.target
target_names = list(iris_dataset.target_names)
dict_map = {i: target_names[i] for i in range(len(target_names))}
# Train - Test Split
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_labels, test_size=0.2, random_state=1, stratify=iris_labels)
K_folds = 4

x = PrettyTable()
models = []
x.field_names = ["Neurons in first hidden layer", "Neurons in second hidden layer", "Activation function",
                 "Training Accuracy", "Validation Accuracy", "Test Accuracy"]
for activ in ["logistic", "tanh", "relu"]:
    for num_of_neurons_1 in [2, 5, 10, 20, 50]:
        for num_of_neurons_2 in [2, 5, 10, 20, 50]:
            model = MLPClassifier(hidden_layer_sizes=(num_of_neurons_1, num_of_neurons_2,), solver='sgd', activation=activ,
                                  learning_rate_init=0.01, max_iter=10000, random_state=0, tol=1e-4)
            model.fit(X_train, y_train)
            models.append(model)
            predict_train = model.predict(X_train)
            scores = cross_val_score(model, X_train, y_train, cv=K_folds)
            predict = model.predict(X_test)
            x.add_row([str(num_of_neurons_1), str(num_of_neurons_2), activ, accuracy_score(predict_train, y_train), str(scores.mean())[:5] + " ± " + str(scores.std())[0:5],
                       str(round(accuracy_score(y_test, predict), 5))])
            print("Finished Training with ", str(num_of_neurons_1), " neurons in first hidden layer, ", str(num_of_neurons_2)
                  , " neurons in second hidden layer and ", activ, " activation function")
print(x)
