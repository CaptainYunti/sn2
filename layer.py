import numpy as np

class Layer:
    def __init__(self, number_neurons, layer_input, learning_rate):
        self.number_neurons = number_neurons
        self.learning_rate = learning_rate
        self.prevl_number_neurons = layer_input.shape[0]
        bound = np.sqrt(1.55/number_neurons)
        self.weights = np.random.uniform(low=-bound, high=bound, size=(number_neurons, self.prevl_number_neurons+1))
        
        

    def forward(self, layer_input) -> np.ndarray:
        self.layer_input = np.vstack(np.array([1]), layer_input)
        self.net = self.weights.dot(self.layer_input)
        
        return self.elu(self.net)
    

    def backward(self, der_loss) -> np.ndarray:
        self.delta = der_loss * self.d_elu(self.net_matrix)
        self.weights -= self.learning_rate * self.delta.dot(self.layer_input.t)

        return der_loss[1:]
    

    
    def elu(self, x) -> np.ndarray:
        if x <= 0:
            x = np.exp(x) - 1
        
        return x
    

    def d_elu(self, x) -> np.ndarray:
        if x > 0:
            y = 1
        else:
            y = np.exp(x)

        return y
    






    
