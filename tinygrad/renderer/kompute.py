from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.ops import GroupOp, Ops, UOp, PatternMatcher, UPat
from tinygrad import dtypes
from tinygrad.dtype import DType, PtrDType, ImageDType
from tinygrad.helpers import strip_parens, getenv, prod, dedup, AMX
import math
import re
from collections import defaultdict, Counter
from typing import Optional, Union, Literal, Callable, cast
from tinygrad import dtypes


glsl_matcher = None

from tinygrad.runtime.ops_kp import *

EXTENSIONS = {
	"half": [
		"#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n",
		"#extension GL_EXT_shader_16bit_storage : require\n"
	]
}

def _bitcast(ctx, x):
	# GLSL is very picky about bitcasting, insofar as it has dedicated
	# bitcasting functions for each data type.
	# A switch to change between them is defined here
	# keys are (in_type, out_type)
	bitcast_functions = {
		(dtypes.float, dtypes.int): "floatBitsToInt",
		(dtypes.double, dtypes.long): "floatBitsToInt",
		(dtypes.half, dtypes.short): "floatBitsToInt",
		
		(dtypes.float, dtypes.uint): "floatBitsToUint",
		(dtypes.double, dtypes.ulong): "floatBitsToUint",
		(dtypes.half, dtypes.ushort): "floatBitsToUint",
		
		(dtypes.int, dtypes.float): "intBitsToFloat",
		(dtypes.long, dtypes.double): "intBitsToFloat",
		(dtypes.short, dtypes.half): "intBitsToFloat",
			
		(dtypes.uint, dtypes.float): "uintBitsToFloat",
		(dtypes.ulong, dtypes.double): "uintBitsToFloat",
		(dtypes.ushort, dtypes.half): "uintBitsToFloat"
	}
	
	# x should have single argument
	in_type = x.src[0].dtype
	out_type = x.dtype
	return f"{bitcast_functions[in_type, out_type]}({ctx[x.src[0]]})"

glsl_rewrite = PatternMatcher([
	(UPat(Ops.DEFINE_ACC, name="x"), lambda ctx,x: ctx[x.src[0]]),
	(UPat(Ops.ASSIGN, name="x"), lambda ctx,x: f"{ctx[x.src[0]]} = {ctx[x.src[1]]};"),
	(UPat(Ops.IF, name="x"), lambda ctx,x: f"if ({ctx[x.src[0]]}) {{"),
	(UPat((Ops.ENDIF, Ops.ENDRANGE)), lambda ctx: "}"),
	(UPat(Ops.WMMA, name="x"), lambda ctx,x: f"__{x.arg[0]}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]})"),
	# r method accesses
	(UPat(Ops.RANGE, name="x"),
	lambda ctx,x: f"for ({ctx.render_dtype(x.dtype)} {ctx[x]} = {ctx[x.src[0]]}; {ctx[x]} < {ctx[x.src[1]]}; {ctx[x]}++) {{"),
	(UPat(Ops.VECTORIZE, name="x"),
	lambda ctx,x: f"{ctx.float4.replace('float4', ctx.render_dtype(x.dtype))}" + \
	(f"{{{','.join([ctx[y] for y in x.src])}}}" if ctx.device in {'CPU', 'DSP'} else f"({','.join([ctx[y] for y in x.src])})")),
	(UPat(Ops.CAST, name="x"), lambda ctx,x:
	f"__builtin_convertvector({ctx[x.src[0]]}, {ctx.render_dtype(x.dtype)})" if x.dtype.count > 1 and not isinstance(x.dtype, PtrDType) else None),
	(UPat(Ops.CAST, name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, ctx[x.src[0]])})"),
	(UPat(Ops.BITCAST, name="x"), _bitcast ),#lambda ctx,x: f"(*(({ctx.buffer_prefix}{ctx.render_dtype(x.dtype)}*)&{ctx[x.src[0]]}))"),
	(UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx,x: f"{ctx.smem_align}{ctx.smem_prefix}{ctx.render_dtype(x.dtype.base)} {ctx[x]}[{x.dtype.size}];"),
	(UPat(Ops.BARRIER), lambda ctx: ctx.barrier),
	(UPat(Ops.NOOP, name="x"), lambda ctx,x: ctx[x.src[0]]),
	(UPat(Ops.SPECIAL, name="x"), lambda ctx,x: f"{ctx.code_for_workitem[x.arg[0][0]](x.arg[0][-1])}; /* {x.arg[1]} */"),
	
	# max should be easies
	(UPat(Ops.MAX, name="m"), lambda ctx, m: f"max({ctx[m.src[0]]}, {ctx[m.src[1]]})"),
	
	# const
	(UPat(Ops.CONST, arg=math.inf, name="x"), lambda ctx, x: f"({ctx.render_cast(x.dtype, ctx.infinity)})"),
	(UPat(Ops.CONST, arg=-math.inf, name="x"), lambda ctx, x: f"({ctx.render_cast(x.dtype, f'-{ctx.infinity}')})"),
	(UPat(Ops.CONST, dtype=dtypes.floats, name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, ctx.nan)})" if math.isnan(x.arg) else None),
	(UPat(Ops.CONST, dtype=dtypes.float, name="x"), lambda ctx,x: f"{x.arg}f"),
	(UPat(Ops.CONST, dtype=dtypes.int64, name="x"), lambda ctx,x: f"{x.arg}ll"),
	(UPat(Ops.CONST, dtype=dtypes.uint64, name="x"), lambda ctx,x: f"{x.arg}ull"),
	(UPat(Ops.CONST, dtype=dtypes.uint32, name="x"), lambda ctx,x: f"{x.arg}u"),
	(UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx,x: "true" if x.arg else "false"),
	# consts are rendered to larger type and casted
	(UPat(Ops.CONST, (dtypes.bfloat16, dtypes.half), name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, f'{x.arg}f')})"),
	(UPat(Ops.CONST, (dtypes.uint8, dtypes.uint16), name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, f'{x.arg}u')})"),
	(UPat(Ops.CONST, (dtypes.int8, dtypes.int16), name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, x.arg)})"),
	# default const render
	(UPat(Ops.CONST, name="x"), lambda ctx,x: str(x.arg)),
	# new load/store
	(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var('idx'))),
		lambda ctx,buf,idx: f"{ctx[buf]}[{strip_parens(ctx[idx]) if idx.arg == Ops.ADD else ctx[idx]}]"),
	(UPat(Ops.LOAD, src=(UPat.var('bidx'), UPat.var("var"), UPat.var("gate"))), lambda ctx,bidx,var,gate: f"({ctx[gate]}?*{ctx[bidx]}:{ctx[var]})"),
	(UPat(Ops.LOAD, src=(UPat.var('bidx'),), allow_any_len=True), lambda ctx,bidx: f"{ctx[bidx]}"),
	(UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var")), allow_any_len=True), lambda ctx,bidx,var: f"{ctx[bidx]} = {ctx[var]};"),
	
	#(UPat(Ops.CMPNE, name = "x"), lambda ctx, x: f"{x}"),
	# alu/gep
	(UPat(GroupOp.ALU, name="x"), lambda ctx,x: ctx.code_for_op[x.op](
		*([strip_parens(ctx[v]) if v.op == x.op and x.op in {Ops.ADD, Ops.MUL, Ops.XOR} else ctx[v] for v in x.src]), x.dtype)),
	(UPat(Ops.GEP, name="x"), lambda ctx,x: ctx[x.src[0]] + \
	(f"[{x.arg[0]}]" if x.src[0].dtype.count > (8 if ctx.device in {"CUDA", "NV"} else 4) or ctx.device in {'CPU', 'DSP'} else \
	 f".{'xyzwabcd'[x.arg[0]]}")),
	# custom passes through with format
	(UPat(Ops.CUSTOM, name="x"), lambda ctx,x: x.arg.format(*[ctx[y] for y in x.src]))
	
	# hopefully these worksies
	# insert a NOOP before BITCAST to force it to be rendered. not needed on all backends?
	
	# devectorize any bools
	#(UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN, Ops.INDEX), dtype=dtypes.bool, name="alu"), no_vectorized_alu),
	# CAST (from bool) can't be vectorized
	#(UPat(Ops.CAST, src=(UPat(dtype=dtypes.bool),), name="alu"), no_vectorized_alu),
	# WHERE can't be vectorized
	#(UPat(Ops.WHERE, name="alu"), no_vectorized_alu),
	
])

class GLSLRenderer(CStyleLanguage):
	device = "KP"
	
	# TODO: every device is going to have a different global and local max,
	# so implement a means of getting that
	global_max = (65535, 65535, 65535)
	local_max = (256, 256, 64)
	
	# how does this work?
	code_for_workitem = {"g": lambda x: f"int( gl_WorkGroupID[{x}] )", "l": lambda x: f"int( gl_LocalInvocationID[{x}] )", "i": lambda x: f"int( gl_GlobalInvocationID[{x}] )"}
	extra_matcher = glsl_matcher
	supports_float4 = False
	#barrier = "workgroupBarrier();"
	# if a return b else return c
	# lets see...
	code_for_op = {**CStyleLanguage.code_for_op,
		#Ops.WHERE: lambda a,b,c,dtype: f"(bool({a})?{b}:{c})",
		#Ops.CMPLT: lambda a,b,dtype: f"bool({a} < {b})"
		#Ops.CMPNE: lambda a,b,dtype, x: f"({a} != {b}))" # this is the most ridiculous way to do it...
	}# f"select({c},{b},{a})"}
	#del code_for_op[Ops.CMPNE]
	nan = "(0.0 / 0.0)"
	infinity = "(1.0 / 0)"
	barrier = "barrier();"
	string_rewrite = glsl_rewrite
	
	"""
	Some types will require extensions to be enabled in order for them to be used.
	
	uchar, char will require:
	#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
	#extension GL_EXT_shader_8bit_storage : require
	
	short, ushort will require:
	#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
	#extension GL_EXT_shader_16bit_storage : require
	
	half will require:
	#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
	#extension GL_EXT_shader_16bit_storage : require
	"""
	type_map = { dtypes.float: "float", dtypes.uchar: "uint8_t", dtypes.ushort: "uint16_t", dtypes.short: "int16_t",
		dtypes.char: "int8_t", dtypes.int32: "int", dtypes.uint32: "uint", dtypes.bool: "bool", dtypes.half: "float16_t" }
	
	def __init__(self, device):
		self._kompute_device = device
		super().__init__()
		# have to have self declared in order to render the dtype I guess
		#self.code_for_op[Ops.ADD] = lambda a, b, dtype: f"( {self.render_dtype(dtype)}({a})  +  {self.render_dtype(dtype)}({b}) )"
		#self.code_for_op[Ops.SUB] = lambda a, b, dtype: f"( {self.render_dtype(dtype)}({a})  -  {self.render_dtype(dtype)}({b}) )"
		#self.code_for_op[Ops.MUL] = lambda a, b, dtype: f"( {self.render_dtype(dtype)}({a})  *  {self.render_dtype(dtype)}({b}) )"
		#self.code_for_op[Ops.FDIV] = lambda a, b, dtype: f"( {self.render_dtype(dtype)}({a})  /  {self.render_dtype(dtype)}({b}) )"
		#self.code_for_op[Ops.IDIV] = lambda a, b, dtype: f"( {self.render_dtype(dtype)}({a})  /  {self.render_dtype(dtype)}({b}) )"
		self.code_for_op[Ops.RECIP] = lambda x,dtype: f"({self.render_dtype(dtype)}(1)/{x})"
	
	def _render_buffer(self, buf):
		name, (dtype, rw) = buf
		dtype_str = self.render_dtype(dtype)
		buffer_idx = re.sub(r'[^\d/]', '', name)
		
		def_str = f"layout(set = 0, binding = {buffer_idx}) buffer buf_{name} {{ {dtype_str} {name}[]; }};\n"
		return def_str
		
	def _render_vulkan_macros(self):
		# this is going to be a giant pain in the ass
		# and now I can kind of see why they don't want to implement a vulkan backend
		# but whatever, onward we go!
		lines = [
			f"#version {self._kompute_device.glsl_version}\n"
			"#ifndef VULKAN\n",
			"#define VULKAN 100\n",
			"#endif\n\n\n",
			"layout(local_size_x = $LOCAL_SIZE_X, local_size_y = $LOCAL_SIZE_Y, local_size_z = $LOCAL_SIZE_Z) in;\n\n"
		]
		return lines
	
	def _extract_uop_params(self, uops, attr):
		for uop in uops:
			yield uop.__getattribute__("dtype")
			for attr_val in self._extract_uop_params(uop.src, attr):
				yield attr_val
	
	def _render_extensions(self, bufs, uops, kernel):
		extensions = []
		extension_keys = []
		for buf in bufs:
			name, (dtype, rw) = buf
			if dtype.base == dtypes.half and (not "half" in extension_keys):
				extensions += EXTENSIONS["half"]
				extension_keys.append("half")
				break
		"""
		print("extracting")
		dtype_list = list( self._extract_uop_params(uops, "dtype") )
		input("extracted")
		print("checking dtypes")
		for dt in dtype_list:
			if dt.base == dtypes.half and (not "half" in extension_keys):
				extensions += EXTENSIONS["half"]
				extension_keys.append("half")
				break
		input("dtypes checked")
		"""
		if (not "half" in extension_keys) and self.render_dtype(dtypes.half) in "\n".join(kernel):
			extensions += EXTENSIONS["half"]
			extension_keys.append("half")
		
		return extensions
		
	def render_dtype(self, dt: DType, mutable = True):
		if isinstance(dt, PtrDType):
			return self.render_dtype(dt.base)
		return super().render_dtype(dt, mutable)
	
	def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
		# I would say implement a different render() method due to the
		# difference between kernel and the overall program,
		# but in Vulkan there is no difference; the shader serves a single purpose.
		#for op in uops: print(op)
		#input("look at the ops you silly")
		
		vulkan_macros = self._render_vulkan_macros()
		
		
		extensions = self._render_extensions(bufs, uops, kernel)
		
		buffer_declarations = []
		for buf in bufs:
			buffer_declarations.append(self._render_buffer(buf) )
		
		#tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(dtype, ImageDType) for _,(dtype,_) in bufs) else ""  # noqa: E501
		tmp = ""
		buftypes = [(name, self.render_dtype(dtype, mutable)+self.buffer_suffix if isinstance(dtype, (ImageDType, PtrDType)) else self.arg_int_prefix if dtype == dtypes.int else None) for name,(dtype,mutable) in bufs]
		
		prg = ''.join(
			vulkan_macros +
			extensions + 
			buffer_declarations +
			[f"{self.kernel_prefix}void {function_name}(",] +
			[") {\n" + tmp] + ['\n'.join(kernel), "\n}"] +
			[f"\nvoid main() {{ {function_name}(); }}"]
		)
		print("float16_t" in prg)
		return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"
		
	def render_cast(self, dt: DType, val: str) -> str:
		return f"{self.render_dtype(dt)}({val})"
	
	def __render(self, uops:list[UOp]) -> str:
		r: dict[UOp, str] = {}
		self.r = r

		child_count = Counter(v for ru in uops for v in ru.src)
		bufs: dict[UOp, tuple[str, tuple[DType, bool]]] = {}
		kernel = []
		depth = 1
		c: defaultdict[str, int] = defaultdict(int)
		name = "test"
		for u in uops:
			if u.op is Ops.NAME:
				name = u.arg
				continue
			if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
				r[u] = f"data{u.arg}" if u.op is Ops.DEFINE_GLOBAL else u.arg[0]
				bufs[u] = (r[u], (u.dtype, False))
				continue

			# mark buffers that we store to writable
			if u.op is Ops.STORE:
				for up in u.src[0].toposort:
					if up.op is Ops.DEFINE_GLOBAL: bufs[up] = (bufs[up][0], (bufs[up][1][0], True))

			# naming
			prefix = None
			if u.op is Ops.SPECIAL:
				r[u] = u.arg[0]
			else:
				prefix = {Ops.RANGE: "ridx", Ops.WMMA: "wmma", Ops.DEFINE_LOCAL: "temp", Ops.CONST: "const",
					Ops.CAST: "cast", Ops.BITCAST: "cast", Ops.GEP: "gep", Ops.VECTORIZE: "cast", Ops.NOOP: "precast",
					Ops.INDEX: "bidx", Ops.DEFINE_ACC: "acc", Ops.LOAD: "val"}.get(u.op, "alu")
				r[u] = f"{prefix}{c[prefix]}"

			l = cast(str, self.string_rewrite.rewrite(u, ctx=self))
			
			if "*" in l and (not "/*" in l):
				pass#print(u.op, l)
				#input("you dun goofded")
			assert l is not None, f"failed to render {u.op} {u.dtype} {[(x.op,x.dtype) for x in u.src]} {u.arg}"

			if u.op in {Ops.ENDIF, Ops.ENDRANGE}: depth -= 1
			if u.op in {Ops.CONST, Ops.GEP, Ops.INDEX, Ops.CUSTOM} or \
					(u.op in {Ops.VECTORIZE, *GroupOp.ALU, Ops.CAST, Ops.BITCAST} and child_count[u] == 1 and not getenv("EXPAND_SSA")):
				r[u] = l
			else:
				if u.op in {Ops.RANGE, Ops.ASSIGN, Ops.DEFINE_LOCAL} or u.dtype == dtypes.void:
					if u.op is Ops.ASSIGN: r[u] = r[u.src[0]]
				else:
					l = f"{self.render_dtype(u.dtype)} {r[u]} = {l}" + (";" if u.op is not Ops.SPECIAL else "")
				kernel.append("  "*depth + l)
				if prefix: c[prefix] += 1  # if it was used, increment
			if u.op in {Ops.IF, Ops.RANGE}: depth += 1
		del self.r

		# NOTE: this relies on bufs dict preserving order
		return self.render_kernel(name, kernel, list(bufs.values()), uops)
