import numpy as np
import json
import matplotlib.pyplot as plt
import gzip
import cv2
from PIL import Image

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
        with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'r') as f:
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
        with open('training/data.json') as f:
            raw_data = json.load(f)
        
        self.training_set = np.array(raw_data['images'])
        self.labels = np.array(raw_data['labels'])

        self.test_set = np.array(raw_data['images_test'])
        self.test_labels = np.array(raw_data['labels_test'])

    def data_preprocessing(self):
        # Scalin data
        self.training_set = (self.training_set.astype(np.float32) - 127.5) / 127.5
        self.test_set = (self.test_set.astype(np.float32) - 127.5) / 127.5
        # print (self.training_set.min(), self.training_set.max())
   
        # Reshaping into the 1 dimensional array
        self.training_set =  self.training_set.reshape(self.training_set.shape[0], self.training_set.shape[1] * self.training_set.shape[2])
        self.test_set =  self.test_set.reshape(self.test_set.shape[0], self.test_set.shape[1] * self.test_set.shape[2])

        # Data Shafelling
        keys = np.array(range(self.training_set.shape[0]))
        np.random.shuffle(keys)
        self.training_set = self.training_set[keys]
        self.labels = self.labels[keys]

        # Labers preporcessing
        self.processed_labels = np.zeros((len(self.training_set), 10))

        for i in range(len(self.training_set)):
            self.processed_labels[i][self.labels[i]] = 1

        # import matplotlib.pyplot as plt
        # plt.imshow((self.test_set[0].reshape(28, 28)))
        # plt.show()

        # print(self.test_labels[0])

    def chunking(self):
        self.batch_size = 32
        self.steps = self.training_set.shape[0] // self.batch_size

        if self.steps * self.batch_size < self.training_set.shape[0]:
            self.steps += 1

    def parameters_generation(self):
        instances = self.training_set.shape[0]
        attributes = self.training_set.shape[1]
        output_labels = len(self.processed_labels[0])

        hidden_nodes1 = int(len(self.training_set[0]) * 1.25)
        
        # hidden_nodes2 = int(len(self.training_set[0]) * 1.25)
        
        # hidden_nodes1 = 500

        self.weights_hidden1 = np.random.rand(attributes, hidden_nodes1) * 0.01
        self.bias_h1 = np.random.randn(hidden_nodes1)

        self.weights_output = np.random.rand(hidden_nodes1, output_labels) * 0.01
        self.bias_o = np.random.randn(output_labels)
        
        print('Parameters generated!')

    def memory(self):
        data = {
            'weights_hidden1': self.weights_hidden1.tolist(),
            'bias_h1': self.bias_h1.tolist(),
            'weights_output': self.weights_output.tolist(),
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
        error_cost = 0
        iter_num = 0

        for i in range(iterations):
            for step in range(self.steps):
                x_batch = self.training_set[step * self.batch_size:(step + 1) * self.batch_size]
                y_batch = self.processed_labels[step * self.batch_size:(step + 1) * self.batch_size]

                ########## Step 1 - Hidden layer #1
                X1 = np.dot(x_batch, self.weights_hidden1) + self.bias_h1
                prediction_h1 = self.sigmoid(X1)

                ########## Step 2 - Output layer
                y = np.dot(prediction_h1, self.weights_output) + self.bias_o
                prediction_o = self.softmax(y)
                
                ######### Back Propagation

                ########## Step 1 - Output layer
                pred_h1 = prediction_h1

                error_cost_o = prediction_o - y_batch

                der_cost_o = np.dot(pred_h1.T, error_cost_o)
                dcost_bo = error_cost_o
                
                ########## Step 2 - Hidden layer #3
                taining_data = x_batch

                weight_o = self.weights_output
                error_cost_h1 = np.dot(error_cost_o, weight_o.T)
                derivative_h1 = self.sigmoid_der(X1)
                pred_h2 = prediction_h1
                der_cost_h1 = np.dot(taining_data.T, derivative_h1 * error_cost_h1)

                dcost_bh1 = error_cost_h1 * derivative_h1

                ########## Update Weights and Biases

                self.weights_hidden1 -= self.learning_rate * der_cost_h1
                self.bias_h1 -= self.learning_rate * dcost_bh1.sum(axis=0)

                self.weights_output -= self.learning_rate * der_cost_o
                self.bias_o -= self.learning_rate * dcost_bo.sum(axis=0)
                
                loss = np.sum(-y_batch * np.log(prediction_o))
                error_cost = loss

            iter_num += 1
            # print('Iterations: ' + str(iter_num))
            # print(error_cost)
        self.memory()
        print('Training is done!')
    
    def process_image(self, im):
        img = np.array(im)

        processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.bitwise_not(processed_image)
        processed_image = cv2.resize(processed_image, (28, 28), interpolation = cv2.INTER_LINEAR)

        processed_image = (processed_image.astype(np.float32) - 127.5) / 127.5
        processed_image =  processed_image.reshape(-1)

        return processed_image

    def think(self, sample):
        np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

        with open('training/params.json') as f:
            raw_data = json.load(f)

        hidden_layer1 = np.array(raw_data['weights_hidden1'])
        output_layer = np.array(raw_data['weights_output'])
        bias_h1 = np.array(raw_data['bias_h1'])
        bias_o = np.array(raw_data['bias_o'])

        X1 = np.dot(sample, hidden_layer1) + bias_h1
        hidden_layer1 = self.sigmoid(X1)

        y = np.dot(hidden_layer1, output_layer) + bias_o
        output_layer = np.array(self.softmax(y))
        
        label = np.argmax(output_layer)
        
        # print(output_layer)
        # print("The number is:")
        # print(label)

        # import matplotlib.pyplot as plt
        # plt.imshow((sample.reshape(28, 28)))
        # plt.show()
        
        return label
        
    def testing(self):
        error = 0
        correct = 0
        count = 0

        for i in range(len(self.test_set)):
            result = self.think(self.test_set[i])
            label = self.test_labels[i]
            
            # print('Result:')
            # print(result)
            # print('Label:')
            # print(label)

            if (result == label):
                correct += 1
            else:
                error += 1
            
            count += 1
            # print(count)

        # Calculating error rate
        error_result = (error / len(self.test_set)) * 100
        print('Error rate is: ' + str(error_result) + '%')

############################### Execution ################################
net = NN()

# Extracting data from Minist database
# net.extracting_data()

net.load_data()
net.data_preprocessing()
net.chunking()
net.parameters_generation()
net.training(1000)
net.testing()

# sample = net.process_image(Image.open(r"C:\Users\User\Desktop\111.png"))

# net.think(sample)