import numpy as np

def new_weigths(weights, entry, error, learning_rate):
    for i in range(len(weights)):
        weights[i] = weights[i] + learning_rate * error * entry[i]
    return weights

def run_perceptron(weights, data, labels, learning_rate=1):
    epoch_error = 0
    # Para cada instancia e label

    for x, y in zip(data, labels):
        # IMPLEMENTE AQUI A ATUALIZACAO DOS PESOS
        result_epoch = np.dot(weights, x)

        if result_epoch <= 0:
            label_obtained = 0
        else:
            label_obtained = 1
        
        if y != label_obtained:
            epoch_error += 1
            error = y - label_obtained
            weights = new_weigths(weights, x, error, learning_rate)
        
    return weights, epoch_error

