import os
from tinygrad.device import Compiler

class GLSLCompiler(Compiler):
	def __init__(self, cachekey = "glsl"):
		super().__init__(cachekey)
	
	def compile(self, src: str) -> bytes:
		# first write the shader to a temporary file
		shader_fn = "tmp.glsl"
		with open(shader_fn, "w") as f:
			f.write(src)
		
		# compile to temp file, then load the SPIR-V bytecode
		spirv_fn = "tmp.spirv"
		if os.path.exists(spirv_fn):
			os.remove(spirv_fn)
		result =os.system(f"glslc -fshader-stage=compute {shader_fn} -o {spirv_fn}")
		#assert os.path.exists(spirv_fn)
		if not os.path.exists(spirv_fn):
			input("read tmp.glsl u dummy")
			raise Exception("COMPILE FAILED")
		with open(spirv_fn, "rb") as f:
			return f.read()
			
	def disassemble(self, lib: bytes):
		raise NotImplementedError
