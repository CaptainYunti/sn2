import mnist_loader
import numpy as np
from net import Net


if __name__ == "__main__":
    
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    #network = Net([784, 512, 256, 128, 10])
    #network = Net([784, 256, 128, 10], learning_rate=0.001)
    network = Net([784, 512, 256, 10], learning_rate=0.001, number_epoch=20)

    network.fit(tuple(training_data), validation_data=tuple(validation_data))

    network.test(tuple(test_data))