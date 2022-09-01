import math
DEBUG = 0


class gtype:
    def __init__(self, data, children=(), op="", label=""):
        self.data = data
        self.prev = set(children)
        self.op = op
        self.label = label
        self.grad = 0.0
        self.backward = lambda: None

    def __repr__(self):
        if self.op == "" or DEBUG == 0:
            return f"Value(data={str(self.data)})"
        else:
            return f"Value(data={str(self.data)},children={self.children},op={self.op})"

    def __add__(self, other):
        if not isinstance(other, gtype):
            other = gtype(other)
        out = gtype(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out.backward = backward
        return out

    def __radd__(self, other):
        if not isinstance(other, gtype):
            other = gtype(other)
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, gtype):
            other = gtype(other)
        return self.__add__(gtype(-1 * other.data))

    def __rsub__(self, other):
        if not isinstance(other, gtype):
            other = gtype(other)
        return other.__add__(gtype(-1 * self.data))

    def __mul__(self, other):
        if not isinstance(other, gtype):
            other = gtype(other)
        out = gtype(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.backward = backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        assert isinstance(other, (int, float)
                          ), "only supporting int/float powers for now"
        out = gtype(self.data**other, (self,), f'**{other}')

        def backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out.backward = backward

        return out

    def tanh(self):
        exp = math.e ** (2*self.data)
        out = gtype((exp-1)/(exp+1), (self,), "tanh")

        def backward():
            self.grad += (1 - out.data**2) * out.grad
        out.backward = backward
        return out

    def full_backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v.backward()
