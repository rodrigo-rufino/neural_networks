import numpy as np
import matplotlib.pyplot as plt
import data.data_manager as data_manager
import util.util_math as util_math
import util.util_data as util_data
import random


class Perceptron:
    dataset = []

    def __init__(self):
        self.bias = -1.0
        self.learning_rate = 0.001
        self.act_function = util_math.sign_function


    def add_bias(self, dataset):
        for row in range(len(dataset)):
            dataset[row] = [self.bias] + dataset[row]
        return dataset


    def train(self, dataset, output):

        x_limit, y_limit = util_data.plot_limits(dataset)

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
                # self.boundary_line(w, x_limit, 'r-')
            print("Epoch = {}, Error = {}\n".format(epoch, error))
            epoch += 1
        # self.boundary_line(w, x_limit, 'k-')
        return w
    

    def test(self, w, data):
        v = np.dot(np.transpose(w), data)
        y = self.act_function(v)
        return y

    
    def boundary_line(self, w, x_max, line_params):
        # para x = 0
        y1 = (-(w[1]*0 + w[0]*self.bias)) / w[2]
        # para x = x_max
        y2 = (-(w[1]*x_max + w[0]*self.bias)) / w[2]

        plt.plot([0, x_max], [y1, y2], line_params)
        

    def boundary_monte_carlo(self, w, max_x, max_y):
        number_of_tests = 1000
        
        for i in range(0, number_of_tests):
            data = [self.bias, random.random() * max_x, random.random() * max_y]
            y = self.test(w, data)
            if y == 1:
                plt.plot(data[1], data[2], 'yo')
            elif y == -1:
                plt.plot(data[1], data[2], 'go')


    def plot_test(self, w, dataset_test, output_test):

        correct = 0
        
        for i in range (0, len(dataset_test)):
            y = self.test(w, dataset_test[i])

            if y == output_test[i]:
                correct += 1
                if y == 1:
                    plt.plot(dataset_test[i][1], dataset_test[i][2], 'bo')
                elif y == -1:
                    plt.plot(dataset_test[i][1], dataset_test[i][2], 'ro')
            else:
                plt.plot(dataset_test[i][1], dataset_test[i][2], 'yx')

        accuracy = 100.0 * float(correct) / float(len(dataset_test))
        print('accuracy: {}/{} ({:.2f}%)'.format(correct, len(dataset_test), accuracy))

        x_limit, y_limit = util_data.plot_limits(dataset_test)
        
        network.boundary_line(w, x_limit, 'k-')
        # network.boundary_monte_carlo(w, x_limit, y_limit)

        axes = plt.gca()
        axes.set_xlim([0,x_limit])
        axes.set_ylim([0,y_limit])


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
    
    network.plot_test(w, dataset_test, output_test)
    plt.show()