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

mat1 = tinygrad.Tensor.randn(4, 5, device = "CPU")
mat2 = tinygrad.Tensor.randn(5, 3, device = "CPU")

out_kp = mat1.to("KP:1").dot(mat2.to("KP:1") )
out_cpu = mat1.dot(mat2)
print(  ( (out_kp.to("CPU") - out_cpu)**2).mean().numpy()  )
