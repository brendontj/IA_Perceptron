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


if __name__ == "__main__":
    DATASET_LIST = [
        "iris",
        "artificial",
        "xor"
    ]

    for DATASET_NAME in DATASET_LIST:
        print("\n\n----\n{}".format(DATASET_NAME))

        if DATASET_NAME == "iris":
            trainX, testX, trainY, testY = load_binary_iris_dataset()

        if DATASET_NAME == "artificial":
            trainX, testX, trainY, testY = load_binary_random_dataset(n_samples=1000, n_features=100)

        if DATASET_NAME == "xor":
            trainX, testX, trainY, testY = load_binary_xor_dataset(cluster_size=500)

        print("Train size: {}".format(len(trainX)))
        print("Test size: {}".format(len(testX)))
        print("Features dimension: {}".format(trainX.shape[1]))

        # As tres linhas a seguir serao utilizadas apenas para visualizar
        # os dados. Se quiser, voce pode comenta-las para executar
        # o algoritmo mais rapidamente
        plt_title = "Training and test set.\n '.' represents "\
                    " training instances and '*' test instances"
        plot_binary_2d_dataset(trainX, testX, trainY, testY, title=plt_title)

        
        # ADICIONE AQUI O CODIGO PARA COMPARAR OS CLASSIFICADORES
