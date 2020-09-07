from sklearn.neural_network import MLPClassifier

# Carrega sua implementacao do perceptron
# Para verificar a implementacao do algoritmo que executa sua funcao, consulte
# o arquivo custom_classifiers.py
from custom_classifiers import PerceptronClassifier

from custom_classifiers import plot_binary_2d_dataset

from custom_classifiers import load_binary_iris_dataset
from custom_classifiers import load_binary_random_dataset
from custom_classifiers import load_binary_xor_dataset

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import csv

if __name__ == "__main__":
    DATASET_LIST = [
        "iris",
        "artificial",
        "xor"
    ]

    execution = []

    for DATASET_NAME in DATASET_LIST:
        print("\n\n----\n{}".format(DATASET_NAME))

        if DATASET_NAME == "iris":
            trainX, testX, trainY, testY = load_binary_iris_dataset()

        if DATASET_NAME == "artificial":
            trainX, testX, trainY, testY = load_binary_random_dataset(n_samples=1000, n_features=100)

        if DATASET_NAME == "xor":
            trainX, testX, trainY, testY = load_binary_xor_dataset(cluster_size=500)
        
        # scaler = StandardScaler()
        # scaler.fit(np.insert(trainX,len(trainX),testX, axis =0))
        # trainX = scaler.transform(trainX)
        # testX = scaler.transform(testX)

        print("Train size: {}".format(len(trainX)))
        print("Test size: {}".format(len(testX)))
        print("Features dimension: {}".format(trainX.shape[1]))

        # As tres linhas a seguir serao utilizadas apenas para visualizar
        # os dados. Se quiser, voce pode comenta-las para executar
        # o algoritmo mais rapidamente
        # plt_title = "Training and test set.\n '.' represents "\
        #             " training instances and '*' test instances"
        # plot_binary_2d_dataset(trainX, testX, trainY, testY, title=plt_title)

        
        # ADICIONE AQUI O CODIGO PARA COMPARAR OS CLASSIFICADORES
        learning_rate_init = [0.001, 0.01, 0.1, 1.0]
        solver = ['lbfgs', 'sgd', 'adam']
        num_iter = [100, 200, 400]
        learning_rate = ['constant', 'invscaling', 'adaptive']
        activation = ['identity', 'logistic', 'tanh', 'relu']

        for l_i in learning_rate_init:
            for n_i in num_iter:
                p_clf = PerceptronClassifier(learning_rate=l_i, max_iter=n_i, verbose=False)
                p_clf.fit(trainX, trainY)
                y_pred = p_clf.predict(testX)

                item = {
                    "dataset_name": DATASET_NAME,
                    "features_dimension": trainX.shape[1],
                    "type_classifier": "PereptronClassifier",
                    "learning_rate_init": l_i,
                    "num_iter": n_i,
                    "score": p_clf.score(testX,testY),
                    "solver": None,
                    "learning_rate": None,
                    "activation": None,
                    "hidden_layer_sizes": None
                }
                execution.append(item)

                for s in solver:
                    for l in learning_rate:
                        for a in activation:
                            clf = MLPClassifier(solver=s, hidden_layer_sizes=(5,), learning_rate_init=l_i, learning_rate=l, activation= a, max_iter=n_i, verbose=False)
                            clf.fit(trainX, trainY)
                            y_pred_mlp = clf.predict_proba(testX)
                            y_aux = np.argmax(y_pred_mlp, 1)
    
                            item = {
                                "dataset_name": DATASET_NAME,
                                "features_dimension": trainX.shape[1],
                                "type_classifier": "MLPClassifier",
                                "learning_rate_init": l_i,
                                "num_iter": n_i,
                                "score": accuracy_score(testY, y_aux),
                                "solver": s,
                                "learning_rate": l,
                                "activation": a,
                                "hidden_layer_sizes": "(5,)"
                            }

                            execution.append(item)

                            clf = MLPClassifier(solver=s, hidden_layer_sizes=(100,), learning_rate_init=l_i, learning_rate=l, activation= a, max_iter=n_i, verbose=False)
                            clf.fit(trainX, trainY)
                            y_pred_mlp = clf.predict_proba(testX)
                            y_aux = np.argmax(y_pred_mlp, 1)

                            item = {
                                "dataset_name": DATASET_NAME,
                                "features_dimension": trainX.shape[1],
                                "type_classifier": "MLPClassifier",
                                "learning_rate_init": l_i,
                                "num_iter": n_i,
                                "score": accuracy_score(testY, y_aux),
                                "solver": s,
                                "learning_rate": l,
                                "activation": a,
                                "hidden_layer_sizes": "(100,)"
                            }

                            execution.append(item)

                            clf = MLPClassifier(solver=s, hidden_layer_sizes=(10,10), learning_rate_init=l_i, learning_rate=l, activation= a, max_iter=n_i, verbose=False)
                            clf.fit(trainX, trainY)
                            y_pred_mlp = clf.predict_proba(testX)
                            y_aux = np.argmax(y_pred_mlp, 1)
                            item = {
                                "dataset_name": DATASET_NAME,
                                "features_dimension": trainX.shape[1],
                                "type_classifier": "MLPClassifier",
                                "learning_rate_init": l_i,
                                "num_iter": n_i,
                                "score": accuracy_score(testY, y_aux),
                                "solver": s,
                                "learning_rate": l,
                                "activation": a,
                                "hidden_layer_sizes": "(10,10)"
                            }

                            execution.append(item)

                            clf = MLPClassifier(solver=s, hidden_layer_sizes=(20,20,20), learning_rate_init=l_i, learning_rate=l, activation= a, max_iter=n_i, verbose=False)
                            clf.fit(trainX, trainY)
                            y_pred_mlp = clf.predict_proba(testX)
                            y_aux = np.argmax(y_pred_mlp, 1)

                            item = {
                                "dataset_name": DATASET_NAME,
                                "features_dimension": trainX.shape[1],
                                "type_classifier": "MLPClassifier",
                                "learning_rate_init": l_i,
                                "num_iter": n_i,
                                "score": accuracy_score(testY, y_aux),
                                "solver": s,
                                "learning_rate": l,
                                "activation": a,
                                "hidden_layer_sizes": "(20,20,20)"
                            }

                            execution.append(item)
        
    with open('results.csv', 'w+', newline='') as csvfile:
        fieldnames= ["dataset_name","features_dimension","type_classifier","learning_rate_init","num_iter","score","solver","learning_rate","activation","hidden_layer_sizes"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for el in execution:
            writer.writerow(el)

