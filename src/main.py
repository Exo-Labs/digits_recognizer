import sys
import numpy as np
import codecs, json
import urllib
import urllib.request
import os
import matplotlib.pyplot as plt

class NN():
    def __init__(self):
        np.set_printoptions(linewidth=200)
        np.random.seed(42)
        self.learning_rate = 10e-4

    def extracting_data(self):
        # Extracting images
        with gzip.open('data/train-images-idx3-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), 'big')
            # third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), 'big')
            # fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), 'big')
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))
        print('Images extraction complete!')

        # Extracting labels
        with gzip.open('data/train-labels-idx1-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of labels
            label_count = int.from_bytes(f.read(4), 'big')
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)
        print('Labes extraction complete!')

        # Extracting test images
        with gzip.open('data/t10k-images-idx3-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), 'big')
            # third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), 'big')
            # fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), 'big')
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            images_test = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))
        print('Test Images extraction complete!')

        # Extracting test labels
        with gzip.open('data/train-labels-idx1-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of labels
            label_count = int.from_bytes(f.read(4), 'big')
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            labels_test = np.frombuffer(label_data, dtype=np.uint8)
        print('Test Labels extraction complete!')

        data = {
                'images': images.tolist(),
                'labels': labels.tolist(),
                'images_test': images_test.tolist(),
                'labels_test': labels_test.tolist(),
            }

        with open('training/data.json', 'w') as json_file:
            json.dump(data, json_file)
        
        print('Data saved!')

    def load_data(self):
        np.set_printoptions(linewidth=200)
        obj_text = codecs.open('training/data.json', 'r', encoding='utf-8').read()
        raw_data = json.loads(obj_text)
        
        self.training_set = np.array(raw_data['images'])
        self.labels = np.array(raw_data['labels'])
        self.test_set = np.array(raw_data['images_test'])
        self.test_labels = np.array(raw_data['labels_test'])

        # import matplotlib.pyplot as plt
        # image = np.asarray(raw_data['images'][0]).squeeze()
        # plt.imshow(image)
        # plt.show()
    
    def data_preprocessing(self):
        # Scalin data
        self.training_set = (self.training_set.astype(np.float32) - 127.5) / 127.5
        self.test_set = (self.test_set.astype(np.float32) - 127.5) / 127.5

        # Reshaping into the 1 dimensional array
        self.training_set =  self.training_set.reshape(self.training_set.shape[0], self.training_set.shape[1] * self.training_set.shape[2])
        self.test_set =  self.test_set.reshape(self.test_set.shape[0], self.test_set.shape[1] * self.test_set.shape[2])
        # print (self.training_set.min(), self.training_set.max())

        # Data Shafelling
        import nnfs
        nnfs.init()

        keys = np.array(range(self.training_set.shape[0]))
        np.random.shuffle(keys)
        self.training_set = self.training_set[keys]
        self.labels = self.labels[keys]

        # import matplotlib.pyplot as plt
        # plt.imshow((self.training_set[5].reshape(28, 28)))
        # print(self.labels[5])
        # plt.show()
        
    def chunking(self):
        self.batch_size = 32
        self.steps = self.training_set.shape[0] // self.batch_size
        if self.steps * self.batch_size < self.training_set.shape[0]:
            self.steps += 1

    def parameters_generation(self):
        x_batch = self.training_set[1 * self.batch_size:(1 + 1) * self.batch_size]
        y_batch = self.labels[1 * self.batch_size:(1 + 1) * self.batch_size]

        # instances = self.training_set.shape[0]
        # attributes = self.training_set.shape[1]
        # output_labels = len(self.labels)

        instances = x_batch.shape[0]
        attributes = x_batch.shape[1]
        output_labels = len(y_batch)

        hidden_nodes = 4

        self.weights_hidden = np.random.rand(attributes, hidden_nodes)
        self.bias_h = np.random.randn(hidden_nodes)
        
        self.weights_output = np.random.rand(hidden_nodes, output_labels)
        self.bias_o = np.random.randn(output_labels)
        print('Parameters generated!')

    def memory(self):
        data = {
            'weights_hidden': self.weights_hidden.tolist(),
            'bias_h': self.bias_h.tolist(),
            'weights_output_h': self.weights_output.tolist(),
            'bias_o': self.bias_o.tolist(),
        }

        with open('training/params.json', 'w') as json_file:
            json.dump(data, json_file)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x) *(1-self.sigmoid (x))

    def softmax(self, x):
        try:
            x.shape[1]
            index = 1
        except IndexError:
            index = 0
        expx = np.exp(x)
        return expx / expx.sum(axis=index, keepdims=True)
    
    def training(self, iterations):
        error_cost = []

        for i in range(iterations):


            for step in range(self.steps):
                ########## Feedforward
                x_batch = self.training_set[step * self.batch_size:(step + 1 ) * self.batch_size]
                y_batch = self.labels[step * self.batch_size:(step + 1 ) * self.batch_size]

                ########## Step 1 - Hidden layer
                X = np.dot(x_batch, self.weights_hidden) + self.bias_h
                prediction_h = self.sigmoid(X)

                ########## Step 2 - Output layer
                y = np.dot(prediction_h, self.weights_output) + self.bias_o
                prediction_o = self.softmax(y)
                ########## Back Propagation

                ########## Step 1 - Output layer
                pred_h = prediction_h

                error_cost_o = prediction_o - y_batch

                der_cost_o = np.dot(pred_h.T, error_cost_o)

                dcost_bo = error_cost_o

                ########## Step 2 - Hidden layer

                weight_o = self.weights_output
                error_cost_h = np.dot(error_cost_o , weight_o.T)
                derivative_h = self.sigmoid_der(X)
                taining_data = x_batch
                der_cost_h = np.dot(taining_data.T, derivative_h * error_cost_h)

                dcost_bh = error_cost_h * derivative_h

                ########## Update Weights and Biases

                self.weights_hidden -= self.learning_rate * der_cost_h
                self.bias_h -= self.learning_rate * dcost_bh.sum(axis=0)

                self.weights_output -= self.learning_rate * der_cost_o
                self.bias_o -= self.learning_rate * dcost_bo.sum(axis=0)

                
                loss = np.sum(-y_batch * np.log(prediction_o))
                print('Loss function value: ', loss)
                error_cost.append(loss)
        
        self.memory()


############################### Execution ################################
net = NN()

# Extracting data from Minist database
# net.extracting_data()

net.load_data()
net.data_preprocessing()
net.chunking()
net.parameters_generation()
net.training(10)