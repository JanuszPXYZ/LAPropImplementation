import numpy as np
from typing import List, Dict
from numpy import ndarray



class LAProp():
    '''Ground up implementation of LAProp algorithm that was proposed by Liu Ziyin, Zhikang T. Wang
    Masahito Ueda in their paper "LAProp: Separating Momentum and Adaptivity in Adam
    
    params:
    learning_rate: float = 0.001
    decay: float = 0.
    mu (also known as beta1 in literature): float = 0.9
    nu (also known as beta2 in literature): float = 0.999
    epsilon: float = 1e-8'''
    def __init__(self,
                 learning_rate = 0.001,
                 decay = 0.,
                 mu = 0.9,
                 nu = 0.999,
                 epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.mu = mu
        self.nu = nu
        self.epsilon = epsilon
        self.timestep = 0


    def decay_applied(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.timestep))

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)


        layer.weight_cache = self.nu * layer.weight_cache + (1 - self.nu) * layer.w_grad ** 2
        layer.bias_cache = self.nu * layer.bias_cache + (1 - self.nu) * layer.b_grad ** 2

        weight_cache_correction = layer.weight_cache / (1 - self.nu ** (self.timestep + 1))
        bias_cache_correction = layer.bias_cache / (1 - self.nu ** (self.timestep + 1))

        layer.weight_momentums = self.mu * layer.weight_momentums + (1 - self.mu) * layer.w_grad / (np.sqrt(weight_cache_correction) + self.epsilon)
        layer.bias_momentums = self.mu * layer.bias_momentums + (1 - self.mu) * layer.b_grad / (np.sqrt(bias_cache_correction) + self.epsilon)

        weight_momentums_correction = layer.weight_momentums / (1 - self.mu ** (self.timestep + 1))
        bias_momentums_correction = layer.bias_momentums / (1 - self.mu ** (self.timestep + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_correction
        layer.biases += -self.current_learning_rate * bias_momentums_correction


    def add_timestep(self):
        self.timestep += 1