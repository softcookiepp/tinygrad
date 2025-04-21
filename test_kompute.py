import tinygrad
from tinygrad import dtypes
import numpy as np
@tinygrad.TinyJit
def test():
	a = tinygrad.Tensor.arange(10, dtype = dtypes.half)
	b = tinygrad.Tensor.randn(10, dtype = dtypes.half)
	d = tinygrad.Tensor.linspace(0, 5, 10, dtype = dtypes.half)
	c = (a + b).sin()*20.0 / (d + 0.01)
	return c.realize()
print(test().numpy() )
pp = tinygrad.Tensor(np.arange(2).astype(np.int8) ).realize()
print(pp.device)

conv = tinygrad.nn.Conv2d(3, 5, 6)
conv.weight.realize()
conv.bias.realize()
input("conv params are goodie")
img = tinygrad.Tensor.randn(2, 3, 256, 266)
print(conv.weight.dtype)
print(conv.bias.dtype)
input()
out = conv(img)
print(out.numpy() )
