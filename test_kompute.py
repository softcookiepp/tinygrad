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

mat1 = tinygrad.Tensor.arange(4*5, device = "CPU", dtype = dtypes.float).reshape(4, 5)
mat2 = tinygrad.Tensor.arange(5*3, device = "CPU", dtype = dtypes.float).reshape(5, 3)

out_cpu = mat1.dot(mat2).realize()
out_kp = mat1.to("KP:1").dot(mat2.to("KP:1") )
print(  ( (out_kp.to("CPU") - out_cpu)**2).mean().numpy()  )
print(out_kp.numpy() )
print(out_cpu.numpy() )

conv = tinygrad.nn.Conv2d(3, 5, 4)
conv.weight.to_("KP:1")
conv.bias.to_("KP:1")
im = tinygrad.Tensor.linspace(0, 10, 2*3*32*32, device="KP:1").reshape(2, 3, 32, 32)
out_im = conv(im).numpy()

conv.weight.to_("CPU")
conv.bias.to_("CPU")
im.to_("CPU")
out_im_cpu = conv(im).numpy()

print( np.mean( (out_im - out_im_cpu)**2 ) )
