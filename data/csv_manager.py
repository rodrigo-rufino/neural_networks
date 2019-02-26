import csv
import os

class CSVManager:
    def __init__(self):
        pass

    def csv_to_list(self, folder, filename):
        csv_file = os.path.join(os.path.dirname(__file__), folder, filename)
        print(csv_file)
        
        dataset = []
        with open(csv_file, 'r') as csv_input:
            csv_reader = csv.reader(csv_input, delimiter=',')
            line = 0
            for row in csv_reader:
                row = [float(i) for i in row]
                dataset.append(row)
        return dataset

    def dataset_input(self, dataset_name):
        return self.csv_to_list(dataset_name, dataset_name + '_input.csv')

    def dataset_output(self, dataset_name):
        return self.csv_to_list(dataset_name, dataset_name + '_output.csv')

if __name__=="__main__":
    csv_manager = CSVManager()
    print(csv_manager.dataset_input('and'))
    pass