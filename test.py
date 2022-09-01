import math
import numpy as np
import matplotlib.pyplot as plt
from mygrad.grad_dtype import gtype
from mygrad.vis_grad import trace, draw_dot

x1 = gtype(2.0, label="x1")
x2 = gtype(0.0, label="x2")
w1 = gtype(-3, label="w1")
w2 = gtype(1.0, label="w2")
b = gtype(6.7, label="b")
x1w1 = x1 * w1
x1w1.label = "x1 * w1"
x2w2 = x2 * w2
x2w2.label = "x2 * w2"
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1w1 + x2w2"
o = x1w1x2w2 + b
o.label = "o"
a = o.tanh()
a.label = "act"
a.full_backward()
draw_dot(a).render(outfile="diagram.png", view=True)
