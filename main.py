from perceptron import Perceptron


def diabetic():
    glucose = [138, 84, 125, 139, 145, 106]
    bmi = [33.6, 38.2, 28.9, 40.7, 44.2, 22.7]
    bias = 1
    diabetic = [1, 0, 1, 0, 1, 0]
    weights = [0.96, 0.37, 0.23]
    learn_rate = 0.1
    epoch = 3

    cls = Perceptron()
    cls.train(glucose, bmi, diabetic, weights, bias, learn_rate, epoch)


def demo():
    x1 = [2, -1, 4]
    x2 = [3, 2, 1]
    weights = [0.4, 0.7, 0.2]
    bias = 1
    labels = [1, 0, 1]
    learn_rate = 0.01
    epoch = 2

    cls = Perceptron()
    cls.train(x1, x2, labels, weights, bias, learn_rate, epoch)


def main():
    choice = True

    if not choice:
        demo()
    else:
        diabetic()


if __name__ == "__main__":
    main()
