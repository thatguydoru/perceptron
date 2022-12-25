from perceptron import OldPerceptron
from perceptron import Perceptron


def new_perceptron():
    # AND data
    train_features = [[0, 0], [0, 1], [1, 0], [1, 1]]
    train_targets = [0, 1, 1, 1]

    cls = Perceptron(2, 2)
    cls.train(train_features, train_targets, 5000)
    preds = cls.predict(train_features)
    print(preds)


new_perceptron()


def diabetic():
    glucose = [138, 84, 125, 139, 145, 106]
    bmi = [33.6, 38.2, 28.9, 40.7, 44.2, 22.7]
    bias = 1
    diabetic = [1, 0, 1, 0, 1, 0]
    weights = [0.96, 0.37, 0.23]
    learn_rate = 0.1
    epoch = 3

    cls = OldPerceptron()
    cls.train(glucose, bmi, diabetic, weights, bias, learn_rate, epoch)


def demo():
    x1 = [2, -1, 4]
    x2 = [3, 2, 1]
    weights = [0.4, 0.7, 0.2]
    bias = 1
    labels = [1, 0, 1]
    learn_rate = 0.01
    epoch = 2

    cls = OldPerceptron()
    cls.train(x1, x2, labels, weights, bias, learn_rate, epoch)
