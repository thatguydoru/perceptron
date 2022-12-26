from smol_vec import Vector
import random


def threshold(v):
    return float(v >= 0)


class Perceptron:
    def __init__(self, input_size, seed=0):
        self.weights = Vector(input_size)

        # random weights and bias
        if seed != 0:
            random.seed(seed)
        self.weights.rand_fill(-1, 1)
        self.bias = random.randint(-1, 1)

        self.learning_rate = 0.1

    def output(self, inputs):
        inputs = Vector.from_array(inputs)
        return threshold(Vector.dot(self.weights, inputs) + self.bias)

    def update(self, inputs, error):
        if error == 0:
            return

        def calc_deltas(x): return self.learning_rate * error * x

        inputs = Vector.from_array(inputs)
        weights_deltas = Vector.static_map(calc_deltas, inputs)
        self.weights.add(weights_deltas)
        self.bias += calc_deltas(self.learning_rate)

    def train(self, x, y, epoch, learning_rate=0.1,  weights=[], bias=None):
        self.learning_rate = learning_rate
        if len(weights) != 0:
            self.weights = weights
        if bias is not None:
            self.bias = bias

        train_data = list(zip(x, y))
        for _ in range(epoch):
            random.shuffle(train_data)
            for features, target in train_data:
                error = target - self.output(features)
                self.update(features, error)

    def predict(self, inputs):
        return [self.output(x) for x in inputs]
