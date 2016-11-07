from logreg import LogReg


class NNet:
    def __init__(self, inp, layer_sizes):
        self.x = inp

        self.layers = []
        prev = self.x
        for i in range(1, len(layer_sizes)):
            new_shape = (layer_sizes[i], layer_sizes[i-1])
            self.layers.append(LogReg(prev, new_shape))
            prev = self.layers[-1].a

        self.a = self.layers[-1].a
        self.params = sum((l.params for l in self.layers), [])
