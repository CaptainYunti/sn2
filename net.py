import numpy as np
from layer import Layer
from tqdm import tqdm
import random

class Net:
    def __init__(self, vec_number_neurons, number_layers=None, number_epoch=10, learning_rate=0.001, soft_max=False, mini_batch=False):
        self.number_layers = number_layers if number_layers is not None else len(vec_number_neurons)
        self.vec_number_neurons = vec_number_neurons
        self.number_epoch = number_epoch
        self.mini_batch = mini_batch
        self.expected_output = None
        self.layers: np.array
        self.prediction: int
        self.accuracy: float

        if not soft_max:
            self.layers = np.array([Layer(vec_number_neurons[i], prevl_number_neurons=vec_number_neurons[i-1], learning_rate=learning_rate)
                      for i in range(1, self.number_layers)])
        else:
            pass
            

    def fit(self, data, number_epoch=None, learning_rate=None, validation_data = None, mini_batch_size = 64):
        if learning_rate is not None:
            for layer in self.layers:
                self.change_learning_rate(layer, learning_rate)

        if number_epoch is not None:
            self.number_epoch = number_epoch

        if self.mini_batch: #not working
            self.mini_batch_learn(data=data, validation_data=validation_data, mini_batch_size=mini_batch_size)

        else:
            self.stochastic_learn(data, validation_data)

    def test(self, data):
        good_predictions = 0
        if len(data) < 1:       
            return

        for sample in tqdm(data, desc=f"Testing: "):
            input, self.expected_output = sample
            if self.predict(sample) == self.expected_output:
                good_predictions += 1

        self.accuracy = np.round(good_predictions / len(data), 2)

        print(f"\nAccuracy: {self.accuracy}")



    def predict(self, sample) -> int:
        input, self.expected_output = sample
        for layer in self.layers:
            output = layer.forward(input)
            input = output

        self.prediction = np.argmax(output)

        return self.prediction

                   


    def change_learning_rate(self, layer: Layer, new_rate):
        layer.change_learning_rate(new_rate)


    def stochastic_learn(self, data:list|tuple, validation_data:tuple):
 
        print("Stochastic learn:")
        for epoch in range(self.number_epoch):
            output = None

            for sample in tqdm(data, desc=f"Epoch {epoch+1}"):
                input, expected_output = sample
                for layer in self.layers:
                    output = layer.forward(input)
                    input = output

                loss_derivative = output - expected_output
                for i in range(self.number_layers-2, 0, -1):
                    loss_derivative = self.layers[i].backward(loss_derivative)

            print('')
            if validation_data is not None:
                self.test(validation_data)
            print('')

        print("End")


    #in progress ...
    def mini_batch_learn(self, data:list|tuple, mini_batch_size, validation_data:tuple):
        if isinstance(data, tuple):
            data = list(data)

        print("Mini-Batch:")
        for epoch in range(self.number_epoch):
            output = None
            random.shuffle(data)
            mini_batch_index = 0

            for sample in tqdm(data, desc=f"Epoch {epoch+1}"):
                input, expected_output = sample
                for layer in self.layers:
                    output = layer.forward(input)
                    input = output

                loss_derivative = output - expected_output
                for i in range(self.number_layers-2, 0, -1):
                    loss_derivative = self.layers[i].backward(loss_derivative)

            print('')
            if validation_data is not None:
                self.test(validation_data)
            print('')
        print("End")

            

    
