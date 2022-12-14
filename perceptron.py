def dot(vec1, vec2):
    sum = 0
    for x, y in zip(vec1, vec2):
        sum += x*y

    return sum


def threshold(v):
    return float(v >= 0)


def output(feat_vec, weights):
    return threshold(dot(feat_vec, weights))


def update(feat_vec, weights, learn_rate, error):
    new_weights = []
    for x, w in zip(feat_vec, weights):
        delta_w = learn_rate * error * x
        new_weights.append(w + delta_w)

    return new_weights


class Perceptron:
    def __init__(self):
        self.weights = []
        self.bias = 0

    def train(self, feat1, feat2, labels, weights, bias, learn_rate, epoch):
        self.weights = weights
        self.bias = bias

        for e in range(epoch):
            print("Epoch:", e)

            for x1, x2, y in zip(feat1, feat2, labels):
                feat_vec = [x1, x2, bias]
                error = y - output(feat_vec, weights)

                print(x1, x2, bias, y, self.weights, error)

                if error <= 0:
                    self.weights = update(
                        feat_vec, self.weights, learn_rate, error)
