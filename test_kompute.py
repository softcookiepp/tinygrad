import tinygrad
from tinygrad import dtypes
import numpy as np
@tinygrad.TinyJit
def test():
	a = tinygrad.Tensor.arange(10, dtype = dtypes.float)
	b = tinygrad.Tensor.randn(10, dtype = dtypes.float)
	d = tinygrad.Tensor.linspace(0, 5, 10, dtype = dtypes.float)
	c = (a + b).sin()*20.0 / (d + 0.01)
	return c.realize()
out = test()
print(out.numpy() )

mat1 = tinygrad.Tensor.arange(4*5, device = "KP:1", dtype = dtypes.float).reshape(4, 5)
mat2 = tinygrad.Tensor.arange(5*3, device = "KP:1", dtype = dtypes.float).reshape(5, 3)

out_kp = mat1.dot(mat2).realize()
input()
out_cpu = mat1.to("CPU").dot(mat2.to("CPU") )
print(  ( (out_kp.to("CPU") - out_cpu)**2).mean().numpy()  )
print(out_kp.numpy() )
print(out_cpu.numpy() )
