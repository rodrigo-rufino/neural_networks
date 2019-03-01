import random

def shuffle_dataset(dataset, output):
    data = []
    for row in range(len(dataset)):
        data.append(dataset[row] + output[row])
    
    random.shuffle(data)

    shuffle_dataset = []
    shuffle_output = []
    for row in data:
        shuffle_dataset.append(row[:-1])
        shuffle_output.append(row[-1])

    return shuffle_dataset, shuffle_output

def plot_limits(dataset):
    x_list, y_list = [], []
    for row in dataset:
        x_list.append(row[1])
        y_list.append(row[2])

    x_limit = max(x_list)+max(x_list)*0.1
    y_limit = max(y_list)+max(y_list)*0.1

    return x_limit, y_limit