import tinygrad
import numpy as np
@tinygrad.TinyJit
def test():
	a = tinygrad.Tensor.arange(10, device = "KP")
	b = tinygrad.Tensor.randn(10, device = "KP")
	d = tinygrad.Tensor.linspace(0, 5, 10, device = "KP")
	c = (a + b).sin()*20 / (d + 0.01)
	return c.realize()
print(test().numpy() )
pp = tinygrad.Tensor(np.arange(2).astype(np.int8), device = "KP").realize()
