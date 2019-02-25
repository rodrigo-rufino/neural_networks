import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    dataset = []

    def __init__(self):
        self.learning_rate = 0.01
        self.act_function = np.sign

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
    dataset = [[-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1], [-1,0.9,0.7], [-1,1,0.7], [-1,0.2,0], [-1,1,0.7]]
    output = [-1,-1,-1,1,1,1,1,1]

    network = Perceptron()
    w = network.train(dataset, output)
    plt.show()

    correct = 0
    for i in range (0, len(dataset)):
        y = network.test(w, dataset[i])
        print("Dataset - {}".format(dataset[i]))
        if y == output[i]:
            correct += 1
            if y == -1: 
                plt.plot(dataset[i][1], dataset[i][2], 'ro')
            else: 
                plt.plot(dataset[i][1], dataset[i][2], 'bo')
    plt.show()
    accuracy = 100.0 * float(correct) / float(len(dataset))
    print('accuracy: {}/{} ({:.2f}%)'.format(correct, len(dataset), accuracy))
