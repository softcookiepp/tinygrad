import functools, struct
import tinygrad
from tinygrad.device import  Compiled, Allocator, Compiler, BufferSpec
from tinygrad.renderer.kompute import GLSLRenderer
from tinygrad.runtime.support.compiler_spirv import GLSLCompiler
from tinygrad.helpers import round_up, OSX
from tinygrad.runtime.autogen import webgpu
from typing import List, Any
import os
import hashlib


# kompute-specific stuff
try:
	import kp
except ImportError:
	raise ImportError("\n	To use the KP backend, the Kompute framework must be installed.\n	To install it, run `pip install kp`")
import numpy as np



def _split_device(device: str):
	idx = 0
	name = device
	if ":" in device:
		name, idx = tuple(device.split(":") )
	return name, int(idx)
	
	
def _get_kp_float_size(nbytes):
	# Kompute stupidly doesn't let you allocate tensors that aren't 32-bit floats??
	assert (nbytes*8) % 32 == 0
	return nbytes*8//32
	
class KomputeAllocator(Allocator):
	def __init__(self, manager: kp.Manager):
		self._mgr = manager
	def _alloc(self, size:int, options:BufferSpec):
		print(size, options)
		fsize = _get_kp_float_size(size)
		print(fsize)
		if options.image is not None:
			raise NotImplementedError
		t = self._mgr.tensor_t(np.random.randn(fsize).astype(np.float32) )
		return self._mgr, t
		
	def _free(self, opaque, options:BufferSpec): pass  # if opaque is a Python object, you don't need a free
	def _copyin(self, dest, src:memoryview):
		mgr, kpt = dest
		src = src.cast("c")
		dest_mv = memoryview(kpt.data() ).cast("c")
		dest_mv[:] = src

	def _copyout(self, dest:memoryview, src):
		mgr, kpt = src
		# first sync with host
		mgr.sequence().eval(kp.OpTensorSyncLocal([kpt]) )
		data = kpt.data()
		mv_src = memoryview(data).cast("c")
		dest = dest.cast("c")
		dest[:] = mv_src

# global compiler, we only need 1!
GLSL_COMPILER = GLSLCompiler()
		
class KomputeProgram:
	def __init__(self, name: str, mgr: kp.Manager, shader_source: str, global_size: tuple, local_size: tuple, vals, wait: bool):
		# only need to save 3 of the parameters? hmmm
		self._name = name
		self._mgr = mgr
		self._shader_source = shader_source
		
	def _local_shader_source(self, local_size):
		# So this is where we are actually going to have to compile the code,
		# because the local size is actually specified in the shader code itself.
		# which is very very dumb :c
		shader_source = self._shader_source
		for ls, s in zip(local_size, ["X", "Y", "Z"]):
			# replace symbolic local size with hardcoded actual local size
			s = f"$LOCAL_SIZE_{s}"
			shader_source = shader_source.replace(s, str(ls) )
		return shader_source
	
	def _get_algorithm(self, bufs, spirv, global_size, vals):
		return KomputeRegistry.get_algorithm(self._mgr,
			spirv, bufs, global_size, vals)
	
	def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
		kp_tensors = []
		print(global_size, local_size)
		
		
		shader_source = self._local_shader_source(local_size)
		spirv = GLSL_COMPILER.compile_cached(shader_source)
		
		for mgr, kpt in bufs:
			# might as well check
			assert mgr == self._mgr
			kp_tensors.append(kpt)
		algo = self._get_algorithm(kp_tensors, spirv, global_size, vals)
		self._mgr.sequence().eval(kp.OpAlgoDispatch(algo) )
		
		
		#raise NotImplementedError



		
class KomputeRuntime:
	def __init__(self, mgr: kp.Manager):
		self._mgr = mgr
	
	def __call__(self, name, shader_source, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
		# Its impossible to tell the local size from here or inside the
		# renderer, so we wait to compile it until later on
		shader_source = shader_source.decode("utf-8")
		return KomputeProgram(name, self._mgr, shader_source, global_size, local_size, vals, wait)


class KomputeRegistry:
	_managers = {}
	_allocators = {}
	_algorithms = {}
	
	@classmethod
	def _initialize_manager(self, device: str):
		backend, idx = _split_device(device)
		assert backend == "KP" # just in case, for now heh
		self._managers[device] = kp.Manager(idx)
		return self._managers[device]
	
	@classmethod
	def get_manager(self, device: str) -> kp.Manager:
		device = tinygrad.Device.canonicalize(device)
		if not device in self._managers.keys():
			self._initialize_manager(device)
		return self._managers[device]
	
	@classmethod
	def get_allocator(self, device: str) -> KomputeAllocator:
		device = tinygrad.Device.canonicalize(device)
		if not device in self._allocators.keys():
			mgr = self.get_manager(device)
			self._allocators[device] = KomputeAllocator(mgr)
		return self._allocators[device]
		
	@classmethod
	def _get_algorithm_signature(self, mgr, spirv, bufs, global_size, vals):
		to_hash = bytes(f"{mgr}{spirv}{list(bufs)}{global_size}{vals}", encoding = "utf-8")
		return hashlib.sha512(to_hash).digest()
		
	@classmethod
	def get_algorithm(self, mgr, spirv, bufs, global_size, vals, hash_key = None):
		if hash_key is None:
			hash_key = self._get_algorithm_signature(mgr, spirv, bufs, global_size, vals)
		if not hash_key in self._algorithms.keys():
			# initialize algorithm
			bufs = list(bufs)
			global_size = list(global_size)
			self._algorithms[hash_key] = mgr.algorithm(bufs, spirv, global_size)
		return self._algorithms[hash_key]
		


class KpDevice(Compiled):
	def __init__(self, device: str):
		device = tinygrad.Device.canonicalize(device)
		self._mgr = KomputeRegistry.get_manager(device)
		
		alloc = KomputeRegistry.get_allocator(device)
		self._runtime = KomputeRuntime(self._mgr)
		super().__init__(device, alloc, GLSLRenderer(self), None, self._runtime)
	
	@property
	def glsl_version(self):
		# Making this a property in case it ends up being hardware-dependent
		return 450

