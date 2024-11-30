import numpy as np

class Layer:
    def __init__(self, number_neurons, layer_input, learning_rate):
        self.number_neurons = number_neurons
        self.learning_rate = learning_rate
        self.prevl_number_neurons = layer_input.shape[0]
        bound = np.sqrt(1.55/number_neurons)
        self.weights = np.random.uniform(low=-bound, high=bound, size=(number_neurons, self.prevl_number_neurons+1))
        
        

    def forward(self, layer_input) -> np.ndarray:
        layer_input = np.vstack(np.array([1]), layer_input)
        output_arg = self.weights*layer_input

        return self.elu(output_arg)
    

    def backward(self, output) -> None:
        matrix = self.weights.T*self.d_elu()
        self.weights -= self.d_elu() * self.learning_rate
    

    
    def elu(self, matrix) -> np.ndarray:
        if matrix <= 0:
            matrix = np.exp(matrix) - 1
        
        return matrix
    

    def d_elu(self, matrix) -> np.ndarray:
        if matrix > 0:
            matrix = 1
        else:
            matrix = np.exp(matrix)

        return matrix
    






    
