from grad_dtype import gtype
from vis_grad import draw_dot
import random


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class neuron(Module):
    def __init__(self, nin):
        self.w = [gtype(random.uniform(-1, 1), label="w") for _ in range(nin)]
        self.b = gtype(random.uniform(-1, 1), label="b")

    def __call__(self, x):  # call with input
        out = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        act = out.tanh()
        act.label = "a"
        return act

    def parameters(self):
        return self.w + [self.b]


class layers(Module):
    def __init__(self, nin, nout):
        self.neurons = [neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params


class MLP(Module):  # Multi layer perceptron - many layers together
    def __init__(self, nin, nouts):  # nout is the size of each layer
        all_layers = [nin] + nouts  # it will be a list of all layers
        self.layers = [layers(all_layers[i], all_layers[i+1])
                       for i in range(len(nouts))]  # already
        # there is one extra layer which is nin. that is why the loop is done for nouts length

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layers in self.layers for p in layers.parameters()]


if __name__ == "__main__":
    #n = neuron(3)
    #x = [1, 2, 3]
    #n = layers(3, 3)
    #n = MLP(3, [2, 1])
    # print(n(x))
    # n(x).full_backward()
    n = MLP(4, [4, 2, 1])
    xs = [
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 0]
    ]
    ys = [0, 5, 1, 0]

    for x in xs:
        print(n(x))

    print("Training")
    for _ in range(5000):
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    #draw_dot(loss).render(outfile="diagram.png", view=True)
        #print("loss:", loss)
        n.zero_grad()
        loss.full_backward()
        for p in n.parameters():
            p.data += (-0.01 * p.grad)

    for x in xs:
        print(n(x))
