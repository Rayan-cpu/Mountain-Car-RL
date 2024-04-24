import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Cat(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def say_hello(self, name):
        pass

class Kitten(Cat):
    def __init__(self):
        super().__init__()

    def bye(self):
        print('bye')
        return    
    
y = Kitten()
y.bye( )