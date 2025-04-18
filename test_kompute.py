import tinygrad
import numpy as np
@tinygrad.TinyJit
def test():
	a = tinygrad.Tensor.arange(10)
	b = tinygrad.Tensor.randn(10)
	d = tinygrad.Tensor.linspace(0, 5, 10)
	c = (a + b).sin()*20 / (d + 0.01)
	return c.realize()
print(test().numpy() )
pp = tinygrad.Tensor(np.arange(2).astype(np.int8) ).realize()
