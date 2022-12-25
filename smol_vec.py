from copy import deepcopy
from random import randint


class Vector:
    """
    Representation of a vector.
    """

    def __init__(self, input_size):
        self.len = 0
        self.data = [0 for _ in range(input_size)]

    def show(self):
        print(self.data)

    def rand_fill(self, start, end):
        """
        Randomly fill the vector.
        """
        self.data = list(map(lambda _: randint(start, end), self.data))

    def map(self, func):
        """
        Apply a function to every element in the vector.
        """
        self.data = list(map(func, self.data))

    def add(self, other):
        self.data = list(map(lambda x, y: x + y, self.data, other.data))

    @staticmethod
    def from_array(arr):
        """
        Make a vector from an array.
        """
        vectored = Vector(len(arr))
        vectored.data = arr

        return vectored

    @staticmethod
    def static_map(func, vector):
        """
        Apply a function to every element in the vector.
        """
        mapped = deepcopy(vector)
        mapped.map(func)

        return mapped

    @staticmethod
    def dot(vector1, vector2):
        """
        Perform a dot product from two vectors.
        """
        product = 0
        for v1, v2 in zip(vector1.data, vector2.data):
            product += (v1 * v2)

        return product
