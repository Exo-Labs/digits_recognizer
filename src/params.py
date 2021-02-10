import numpy as np

class Parameters():
    def __init__(self):
        self.learning_rate = 10e-4

    def parameters_generation(self, layers):
        
        instances = 300
        attributes = 50
        output_labels = 2
        
        hidden_nodes = int(attributes * 0.7)
        
        self.paremeters = {}

        for i in range(layers):
            self.paremeters['w' + str(i + 1)] = np.random.randn(attributes, hidden_nodes) * 0.01
            self.paremeters['b' + str(i + 1)] = np.random.randn(hidden_nodes)
            hidden_nodes = int(hidden_nodes * 0.7)
        
        self.paremeters['wo'] = np.random.randn(hidden_nodes, output_labels) * 0.01
        self.paremeters['bo'] = np.random.randn(output_labels)
        return self.paremeters