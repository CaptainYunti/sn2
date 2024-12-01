import numpy as np

class Layer:
    def __init__(self, number_neurons, prevl_number_neurons, learning_rate):
        self.number_neurons = number_neurons
        self.learning_rate = learning_rate
        self.prevl_number_neurons = prevl_number_neurons
        bound = np.sqrt(1.55/number_neurons)
        self.weights = np.random.uniform(low=-bound, high=bound, size=(number_neurons, self.prevl_number_neurons+1))
        self.delta: np.ndarray
        self.net: np.ndarray
        self.layer_input: np.ndarray
        self.der_loss: np.ndarray
        
        

    def forward(self, layer_input) -> np.ndarray:
        self.layer_input = np.vstack((np.array([1]), layer_input))
        self.net = self.weights.dot(self.layer_input)
        
        return self.elu(self.net)
    

    def backward(self, der_loss) -> np.ndarray:
        self.delta = der_loss * self.d_elu(self.net)
        self.weights -= self.learning_rate * self.delta.dot(self.layer_input.T)
        self.der_loss = self.weights.T.dot(self.delta)

        return self.der_loss[1:]
    

    
    def elu(self, x:np.ndarray) -> np.ndarray:
        x = np.where(x <= 0, np.exp(x) - 1, x)
    
        return x
    

    def d_elu(self, x:np.ndarray) -> np.ndarray:
        y = np.where(x > 0, 1, np.exp(x))

        return y
    

    def change_learning_rate(self, learning_rate) -> None:
        self.learning_rate = learning_rate
    






    
