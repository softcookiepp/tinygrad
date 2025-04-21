import tinygrad
from tinygrad import dtypes
import numpy as np
@tinygrad.TinyJit
def test():
	a = tinygrad.Tensor.arange(10, dtype = dtypes.half)
	b = tinygrad.Tensor.randn(10, dtype = dtypes.half)
	d = tinygrad.Tensor.linspace(0, 5, 10, dtype = dtypes.half)
	print(a.dtype, b.dtype, d.dtype)
	input()
	c = (a + b).sin()*20.0 / (d + 0.01)
	return c.realize()
print(test().numpy() )
pp = tinygrad.Tensor(np.arange(2).astype(np.int8) ).realize()
