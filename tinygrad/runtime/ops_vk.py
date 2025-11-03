#from __future__ import annotations
from typing import Optional, cast
import ctypes, functools, hashlib
from tinygrad.helpers import init_c_var, to_char_p_p, from_mv, OSX, DEBUG, getenv, mv_address
from tinygrad.renderer.clspv import ClspvRenderer
from tinygrad.renderer.glsl import GLSLRenderer
from tinygrad.device import BufferSpec, LRUAllocator, Compiled, Compiler, CompileError, Allocator
import pyclspv
import os
import pytart
import numpy as np
import json
import time
import uuid

# whether or not save the rendered kernels for testing purposes
SAVE_RENDERED_KERNELS = True if "VK_SAVE_RENDERED" in os.environ.keys() else False

# to prevent device timeout
MAX_SUBMISSIONS = 2

USED_RENDERER = "glsl"
if "VK_RENDERER" in os.environ.keys():
	USED_RENDERER = os.environ["VK_RENDERER"]
tart = pytart.Instance()

class VkProgram:
	def __init__(self, device, name, lib):
		self._device = device
		self._name = name
		self.name = name
		self._module = device.get_device().load_shader(lib)
		if USED_RENDERER == "clc":
			self._cl_prg = device.get_device().create_cl_program(self._module)
		else:
			self._pipelines = {}
	
	def _get_pipeline(self, name, local_size, vals = None):
		if USED_RENDERER == "clc":
			if vals is None:
				return self._cl_prg.get_pipeline(name, local_size)
			else:
				return self._cl_prg.get_pipeline(name, local_size, vals)
		else:
			pipeline_key = name + str(local_size) + str(vals)
			if not pipeline_key in self._pipelines.keys():
				if vals is None:
					args = (self._module, "main", None)
				else:
					args = (self._module, "main", None, vals)
				try:
					self._pipelines[pipeline_key] = self._device.get_device().create_pipeline(*args)
				except ValueError:
					input(vals)
			return self._pipelines[pipeline_key]
		
	def __call__(self, *bufs, global_size, local_size, vals = (), wait = False):
		if SAVE_RENDERED_KERNELS:
			try:
				with open("rendered/global_size_table.json", "r") as f:
					global_sizes = json.load(f)
			except:
				global_sizes = {}
			if not self._name in global_sizes.keys():
				global_sizes[self._name] = global_size
				with open("rendered/global_size_table.json", "w") as f:
					json.dump(global_sizes, f)
		bufs = [buf[0] for buf in bufs]
		sequence = self._device.get_device().create_sequence()
		#sequence = self._device.get_sequence()
		vals = np.array(vals, dtype = np.int32)
		if USED_RENDERER is "glsl" and self._device.get_device().metadata.bda:
			# add buffer addresses to the array
			addrs = np.array([buf.address for buf in bufs], dtype = np.uint64)
			vals = np.concatenate( (addrs.view(np.uint32), vals) )
		if len(vals) > 0:
			vals = np.array(vals, dtype = np.int32)
			pipeline = self._get_pipeline(self._name, local_size, vals)
			sequence.record_pipeline(pipeline, global_size, bufs[0:pipeline.num_buffer_args], vals)
			
		else:
			pipeline = self._get_pipeline(self._name, local_size)
			sequence.record_pipeline(pipeline, global_size, bufs[0:pipeline.num_buffer_args])
		# syncing before might be better c:
		self._device.sync()
		self._device.submit_sequence(sequence)
		if wait:
			#self._device.submit_sequence(sequence)
			self._device.sync()
			return 0.0

class ClspvCompiler(Compiler):
	def __init__(self, device, compile_key: str):
		self._device = device
		super().__init__(f"compile_clspv_{compile_key}")
	def compile(self, src: str):
		src += "\n" # clspv will complain about missing newlines
		spv = pyclspv.compile(src)
		shader_module = self._device.get_device().load_shader(spv)
		return shader_module.spv
	
	@property
	def __name__(self):
		return "VK"

class GLSLCompiler(Compiler):
	def __init__(self, device, compile_key: str):
		self._device = device
		super().__init__(f"compile_glsl_{compile_key}")
	def compile(self, src: str):
		src += "\n" # clspv will complain about missing newlines
		shader_module = self._device.get_device().compile_glsl(src)
		self._device.add_src(src)
		return shader_module.spv
	
	@property
	def __name__(self):
		return "VK"
		
class VkAllocator(LRUAllocator['VkDevice']):
	def _alloc(self, size, options):
		self.dev.sync()
		start = time.monotonic()
		buf = self.dev.get_device().allocate_buffer(size), options
		#print(f"allocation took {(time.monotonic() - start)*1000} ms")
		return buf
	def _free(self, opaque, options):
		self.dev.sync()
		start = time.monotonic()
		buf = self.dev.get_device().deallocate_buffer(opaque[0])
		#print(f"free took {(time.monotonic() - start)*1000} ms")
		return buf
	def _copyin(self, dst, src: memoryview):
		self.dev.sync()
		start = time.monotonic()
		dst[0].copy_in(src)
		#print(f"copyin took {(time.monotonic() - start)*1000} ms")
	def _copyout(self, dst: memoryview, src):
		self.dev.sync()
		start = time.monotonic()
		src[0].copy_out(dst)
		#print(f"copyout took {(time.monotonic() - start)*1000} ms")

class VkDevice(Compiled):
	def __init__(self, device_ = ""):
		idx = int(device_.split(":")[1]) if len(device_.split(":")) > 1 else 0
		self._device = tart.create_device(idx)
		self._sequence = None
		if USED_RENDERER == "clc":
			compiler = functools.partial(ClspvCompiler, self, f"compile_tart_cl_{device_}")
			renderer = ClspvRenderer
		elif USED_RENDERER == "glsl":
			compiler = functools.partial(GLSLCompiler, self, f"compile_tart_glsl_{device_}")
			renderer = functools.partial(GLSLRenderer, self._device)
		else:
			raise NotImplementedError
		super().__init__(device_, VkAllocator(self), [(renderer, compiler)], functools.partial(VkProgram, self))
		self._submission_counter = 0
		self._kernel_sources = []
		self._collect_sources = False
	
	def enable_source_recording(self):
		self._collect_sources = True
	
	def add_src(self, src: str):
		if self._collect_sources:
			self._kernel_sources.append(src)
	
	def get_src(self):
		return self._kernel_sources
	
	def clear_src(self):
		self._kernel_sources = []
	
	def get_device(self):
		return self._device
		
	def get_sequence(self):
		if self._sequence is None:
			self._sequence = self._device.create_sequence()
		return self._sequence
	
	def submit_sequence(self, sequence):
		if self._submission_counter >= MAX_SUBMISSIONS:
			self.sync()
		self._device.submit_sequence(sequence)
		self._submission_counter += 1
	
	def sync(self):
		self._device.sync()
		self._submission_counter = 0
	
	def supports_dtype(self, dtype):
		raise NotImplementedError
