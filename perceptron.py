import numpy as np
import matplotlib.pyplot as plt
import data.data_manager as data_manager
import util.util_math as util_math
import util.util_data as util_data

class Perceptron:
    dataset = []

    def __init__(self):
        self.bias = -1.0
        self.learning_rate = 0.0001
        self.act_function = util_math.sign_function

    def add_bias(self, dataset):
        for row in range(len(dataset)):
            dataset[row] = [self.bias] + dataset[row]
        return dataset

    def train(self, dataset, output):
        w = np.random.rand(len(dataset[0]))
        k = len(dataset)

        epoch = 0
        error = True

        while error:
            error = False      
            for i in range(0, k):
                v = np.dot(np.transpose(w), dataset[i])
                y = self.act_function(v)
                if y != output[i]:
                    w = np.add(w, np.multiply(dataset[i], self.learning_rate * (output[i] - y)))
                    error = True
            print("Epoch = {}, Error = {}\n".format(epoch, error))
            epoch += 1

        return w

    def test(self, w, data):
        v = np.dot(np.transpose(w), data)
        y = self.act_function(v)
        return y


if __name__=="__main__":
    data_manager = data_manager.DataManager()
    network = Perceptron()

    dataset_name = 'iris'

    dataset, output = util_data.shuffle_dataset(data_manager.dataset_input(dataset_name),
                                                data_manager.dataset_output(dataset_name))

    dataset = network.add_bias(dataset)
    dataset_half = int(len(dataset)/2)

    dataset, dataset_test = dataset[:dataset_half], dataset[-dataset_half:]
    output, output_test = output[:dataset_half], output[-dataset_half:]

    w = network.train(dataset, output)
    plt.show()

    correct = 0
    for i in range (0, len(dataset_test)):
        y = network.test(w, dataset_test[i])
        # print("Dataset - {}".format(dataset_test[i]))
        if y == output_test[i]:
            correct += 1
            if y == -1: 
                plt.plot(dataset_test[i][1], dataset_test[i][2], 'ro')
            else: 
                plt.plot(dataset_test[i][1], dataset_test[i][2], 'bo')
    
    accuracy = 100.0 * float(correct) / float(len(dataset_test))
    print('accuracy: {}/{} ({:.2f}%)'.format(correct, len(dataset_test), accuracy))

    plt.show()
