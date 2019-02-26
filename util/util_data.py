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