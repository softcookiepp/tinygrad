from tinygrad.dtype import DType, PtrDType, dtypes, AddrSpace, truncate
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, GroupOp
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite, extra_pm, wmma_args
from tinygrad.helpers import strip_parens
import functools
import os
import numpy as np
from tinygrad.codegen.opt.tc import TensorCore
SAVE_RENDERED_KERNELS = True if "VK_SAVE_RENDERED" in os.environ.keys() else False

if False:
	dummy_cores = [TensorCore(dims=(4,4,1), threads=1, elements_per_thread=(4,4,4*4), dtype_in=dt, dtype_out=dt,
					  swizzle=(((), ('u0', 'u1', 'u2', 'u3'), ()),
							   ((), ('u0', 'u1', 'u2', 'u3'), ())),
					  opts=("u0","u0", "u1", "u1")) for dt,sz in [(dt, 16 // dt.itemsize) for dt in [dtypes.float]]]
elif False:
	dummy_cores = [TensorCore(dims=(2,2,1), threads=1, elements_per_thread=(2,2,4), dtype_in=dt, dtype_out=dt,
		swizzle=(((), ("u0", "u1"), ()),
			   ((), ("u0", "u1"), ())),
		opts=("u0", "u1")) for dt,sz in [(dt, 16 // dt.itemsize) for dt in [dtypes.float]]]
else:
	# this one currently works 100%
	dummy_cores = [TensorCore(dims=(1,1,1), threads=1, elements_per_thread=(1,1,1), dtype_in=dt, dtype_out=dt,
		swizzle=(((), (), ()),
			   ((), (), ())),
		opts=()) for dt,sz in [(dt, 16 // dt.itemsize) for dt in [dtypes.float]]]

def render_store(ctx, b, v):
	if b.op == Ops.CAST:
		if b.dtype.addrspace == AddrSpace.REG or b.dtype.addrspace == AddrSpace.LOCAL:
			idx = b.src[0].src[1]
			buf = b.src[0].src[0]
			return f"STORE_VEC({ctx[buf]}, {ctx[idx]}, {ctx[v]}, {b.dtype.base.count}, {ctx.render_dtype(b.dtype.base)})"
		return f"{ctx._render_cast_index(b)} = {ctx[v]};"
	if b.src[0].op == Ops.DEFINE_REG:
		return f"{ctx[b]} = {ctx.type_map[b.dtype.base]}({ctx[v]});"
	elif b.dtype.base == dtypes.bool:
		return f"{ctx[b]} = {ctx.buf_map(b.dtype)}({ctx[v]});"
	return f"{ctx[b]} = {ctx[v]};"

def _bitcast(ctx, x):
	# GLSL has dedicated bitcasting functions for each data type.
	bitcast_str_map = {
		dtypes.short: "Int16",
		dtypes.int: "Int",
		dtypes.long: "Int64",
		dtypes.ushort: "Uint16",
		dtypes.uint: "Uint",
		dtypes.ulong: "Uint64",
		dtypes.half: "Half",
		dtypes.float: "Float",
		dtypes.double: "Double"
	}
	in_type = x.src[0].dtype.base
	out_type = x.dtype.base
	if (in_type in dtypes.uints or in_type in dtypes.sints) and (out_type in dtypes.uints or out_type in dtypes.sints):
		return f"{ctx.render_dtype(out_type)}({ctx[x.src[0]]})"
	return f"{bitcast_str_map[in_type].lower()}BitsTo{bitcast_str_map[out_type]}({ctx[x.src[0]]})"

def _render_literal(ctx, x):
	if x.dtype == dtypes.uint and x.arg > x.dtype.max:
		xarg = np.array([int(x.arg)], dtype = np.uint64).view(np.uint32)
		return f"uint(pack64(uvec2({xarg[0]}, {xarg[1]})) )"
	return f"({x.arg})"

def _render_fma(ctx, a, b, c):
	if not (a.dtype.base == b.dtype.base == c.dtype.base): return None
	if not a.dtype.base.scalar() in dtypes.floats: return None
	return f"fma( {ctx.render_dtype(a.dtype)}({ctx[a.src[0]]}), {ctx.render_dtype(a.dtype)}({ctx[a.src[1]]}), {ctx.render_dtype(a.dtype)}({ctx[b]}) )"

def _render_dot(ctx, a, b):
	if not (a.dtype == b.dtype) or not a.dtype in dtypes.floats: return None
	dt = ctx.render_dtype(a.dtype.vec(2))
	return f"dot( {dt}({ctx[a.src[0]]}, {ctx[b.src[0]]}), {dt}({ctx[a.src[1]]}, {ctx[b.src[1]]}) )"

def _render_index(ctx, idx_uop, b,idx):
	if isinstance(b.dtype, PtrDType) and b.op == Ops.DEFINE_GLOBAL and ctx.supports_float4:
		ctx.add_index_op(b, idx_uop)
		#ctx._index_ops.add(idx_uop)
		#return f"push.{ctx[b]}.data[{strip_parens(ctx[idx]) if idx.arg is Ops.ADD else ctx[idx]}]"
		return f"{ctx.render_bda(b.dtype.base)}(push.{ctx[b]} + {strip_parens(ctx[idx]) if idx.arg is Ops.ADD else ctx[idx]}*{b.dtype.base.itemsize}).data[0]"
	return f"{ctx[b]}[{strip_parens(ctx[idx]) if idx.arg is Ops.ADD else ctx[idx]}]"

glsl_matcher = PatternMatcher([
	(UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx")), allow_any_len=True),
		lambda ctx, b, idx: UOp(Ops.INDEX, dtypes.int64, (b, idx.cast(dtypes.int64)) ) if b.dtype.size > idx.dtype.max else None),
		
])

class GLSLRenderer(CStyleLanguage):
	device = "VK"
	global_max = (2147483647, 65535, 65535)
	local_max = (1024, 1024, 64)
	code_for_workitem = {"g": lambda x: f"int(gl_WorkGroupID.{'xyz'[int(x)]})", "l": lambda x: f"int(gl_LocalInvocationID.{'xyz'[int(x)]})", "i": lambda x: f"int(gl_GlobalInvocationID.{'xyz'[int(x)]})"}
	type_map = { dtypes.double: "double", dtypes.float: "float", dtypes.uchar: "uint8_t", dtypes.ushort: "uint16_t", dtypes.short: "int16_t",
		dtypes.char: "int8_t", dtypes.int32: "int", dtypes.int64: "int64_t", dtypes.uint64: "uint64_t", dtypes.uint32: "uint", dtypes.uint64: "uint64_t", dtypes.bool: "bool", dtypes.half: "float16_t",
		dtypes.float.vec(2): "vec2", dtypes.float.vec(4): "vec4", dtypes.half.vec(2): "f16vec2", dtypes.half.vec(4): "f16vec4", dtypes.int.vec(2): "ivec2", dtypes.int.vec(4): "ivec4",
		dtypes.uint.vec(2): "uvec2", dtypes.uint.vec(4): "uvec4", dtypes.bool.vec(2): "bvec2", dtypes.bool.vec(4): "bvec4",
		dtypes.float.vec(16): "vec16"}
	barrier = "barrier();"
	supports_float4 = False
	float4 = "vec4"
	code_for_op = {**CStyleLanguage.code_for_op, Ops.EXP2: lambda x,dtype: f"exp2_precise({x})",
		Ops.LOG2: lambda x,dtype: f"log2_precise({x})"}
	name = "glsl"
	tensor_cores = dummy_cores
	
	string_rewrite = PatternMatcher([
		(UPat(Ops.WMMA, name="x"), lambda ctx,x: f"{x.arg[0]}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]})"),
		(UPat(Ops.ADD, src = (UPat(Ops.MUL, name = "a"), UPat(Ops.MUL, name = "b"))), _render_dot),
		(UPat(Ops.ADD, src = (UPat(Ops.MUL, name = "a"), UPat.var("b") ), name = "c" ), _render_fma),
		(UPat(Ops.CONST, dtype=dtypes.int64, name="x"), lambda ctx,x: f"{x.arg}l"),
		(UPat(Ops.CONST, dtype=dtypes.uint64, name="x"), lambda ctx,x: f"{truncate[x.dtype](x.arg)}ul"),
		(UPat(Ops.CONST, dtype = dtypes.uint32, name = "x"), _render_literal),
		(UPat.cvar("x", dtype=dtypes.bool), lambda x: "true" if x.arg else "false"),
		(UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx,x:
			f"shared {ctx.buf_map(x.dtype)} {ctx[x]}[{x.dtype.size}];"),
		(UPat(Ops.DEFINE_REG, name="x"), lambda ctx,x:
			f"{ctx.type_map[x.dtype.base]} {ctx[x]}[{x.dtype.size}];"),
		(UPat(Ops.BITCAST, name="x"), lambda ctx,x: _bitcast(ctx, x)),
		(UPat.load(UPat.var("b"), UPat.cvar("v")),lambda ctx,b,v: f"{ctx[b.src[2]]} ? {ctx.render_complete_load(b)} : {ctx[v]}"),
		(UPat(Ops.LOAD, src = (UPat.var("b"),) ), lambda ctx, b: ctx.render_complete_load(b) ),
		(UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(), UPat(), UPat.var("gate"))).or_casted("bidx"), UPat.var("var"))),
			lambda ctx, bidx, var, gate: f"({ctx[gate]}?{ctx.render_complete_load(bidx)}:{ctx.render_complete_load(var)})"),
		(UPat(Ops.LOAD, src=(UPat.var('bidx'),), allow_any_len=True),
			lambda ctx, bidx: ctx.render_complete_load(bidx) if bidx.dtype.base.count > 1 else ctx[bidx]),
		
		# GLSL can't index with signed long integers; they must be unsigned
		(UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx", dtype = dtypes.long)), allow_any_len=True),
			lambda ctx,b,idx: f"{ctx[b]}[uint64_t({strip_parens(ctx[idx]) if idx.arg is Ops.ADD else ctx[idx]})]"),
		(UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx")), allow_any_len=True, name = "idx_uop"), _render_index),
		(UPat.store(UPat.var("b"), UPat.var("v"), allow_any_len=True),
			render_store),
		(UPat(Ops.OR, dtype = dtypes.bool, src = (UPat.var("a", dtype = dtypes.bool), UPat.var("b", dtype = dtypes.bool) )),
			lambda ctx, a, b: f"{ctx.render_dtype(dtypes.bool)}(uint({ctx[a]}) | uint({ctx[b]}))"),
		(UPat(Ops.XOR, dtype = dtypes.bool, src = (UPat.var("a", dtype = dtypes.bool), UPat.var("b", dtype = dtypes.bool) )),
			lambda ctx, a, b: f"{ctx.render_dtype(dtypes.bool)}(uint({ctx[a]}) ^ uint({ctx[b]}))"),
		(UPat(Ops.AND, dtype = dtypes.bool, src = (UPat.var("a", dtype = dtypes.bool), UPat.var("b", dtype = dtypes.bool) )),
			lambda ctx, a, b: f"{ctx.render_dtype(dtypes.bool)}(uint({ctx[a]}) & uint({ctx[b]}))"),
		
		(UPat(Ops.CONST, dtype = dtypes.int, name="x"), lambda ctx,x: f"({x.arg})"),
		(UPat(Ops.CONST, dtype = dtypes.uint, name="x"), lambda ctx,x: f"({x.arg})"),
		
		(UPat(Ops.WHERE, src = (UPat.var("a"), UPat.var("b"), UPat.var("c") )), lambda ctx, a, b, c: f"{ctx[a]} ? {ctx[b]} : {ctx[c]}" ),
		(UPat(Ops.GEP, src = (UPat.var("a", dtype = dtypes.float.vec(16)),), name="x"), lambda ctx,x, a: ctx[x.src[0]] + f".data[{x.arg[0]}]"),
		(UPat(Ops.GEP, name="x"), lambda ctx,x: ctx[x.src[0]] + f"[{x.arg[0]}]"),
		(UPat(Ops.VECTORIZE, dtype = dtypes.float.vec(16), name="x"), lambda ctx, x: f"make_{ctx.render_dtype(x.dtype)}" + f"{ctx.float4_style[0]}{','.join([ctx[y] for y in x.src])}{ctx.float4_style[1]}"),
		(UPat(Ops.VECTORIZE, name="x"), lambda ctx, x: f"{ctx.render_dtype(x.dtype)}" + f"{ctx.float4_style[0]}{','.join([ctx[y] for y in x.src])}{ctx.float4_style[1]}"),
		(UPat(Ops.CMPLT, dtype = dtypes.bool, src = (UPat.var("x", dtype = dtypes.bool), UPat.var("y", dtype = dtypes.bool) ) ),
			lambda ctx, x, y: f"uint({ctx[x]}) < uint({ctx[y]})"),
		(UPat(Ops.RECIPROCAL, src = (UPat.var("x")) ), lambda ctx, x: f"{ctx.render_dtype(x.dtype)}(1) / {ctx[x]}"),
		(UPat(Ops.ADD, dtype = dtypes.float.vec(2), src = (UPat.var("a"), UPat.var("b")) ), lambda ctx, a, b:  f"{ctx[a]} + {ctx[b]}"),
		(UPat(Ops.SUB, dtype = dtypes.float.vec(2), src = (UPat.var("a"), UPat.var("b")) ), lambda ctx, a, b:  f"{ctx[a]} - {ctx[b]}"),
		(UPat(Ops.MUL, dtype = dtypes.float.vec(2), src = (UPat.var("a"), UPat.var("b")) ), lambda ctx, a, b:  f"{ctx[a]} * {ctx[b]}"),
		(UPat(Ops.FDIV, dtype = dtypes.float.vec(2), src = (UPat.var("a"), UPat.var("b")) ), lambda ctx, a, b:  f"{ctx[a]} / {ctx[b]}"),
		(UPat(Ops.ADD, dtype = dtypes.float.vec(4), src = (UPat.var("a"), UPat.var("b")) ), lambda ctx, a, b:  f"{ctx[a]} + {ctx[b]}"),
		(UPat(Ops.SUB, dtype = dtypes.float.vec(4), src = (UPat.var("a"), UPat.var("b")) ), lambda ctx, a, b:  f"{ctx[a]} - {ctx[b]}"),
		(UPat(Ops.MUL, dtype = dtypes.float.vec(4), src = (UPat.var("a"), UPat.var("b")) ), lambda ctx, a, b:  f"{ctx[a]} * {ctx[b]}"),
		(UPat(Ops.FDIV, dtype = dtypes.float.vec(4), src = (UPat.var("a"), UPat.var("b")) ), lambda ctx, a, b:  f"{ctx[a]} / {ctx[b]}"),
	]) + base_rewrite
	
	extra_matcher = glsl_matcher + CStyleLanguage.extra_matcher
	
	def __init__(self, vkdev):
		self._vkdev = vkdev
		# we could manually construct a new dictionary, or we could just do this!
		for key in (Ops.SIN,):
			del self.code_for_op[key]
		# in order to do pointer casting, etc. VK_KHR_buffer_device_address support is required (tart takes care of detecting this)
		self.supports_float4 = vkdev.metadata.bda and vkdev.metadata.ulong
		self._index_ops = {}
		
	def render_cache_record(self, b):
		raise NotImplementedError
	
	def add_index_op(self, b, idx_op):
		if not b in self._index_ops.keys():
			self._index_ops[b] = set()
		# so how do we know how big a chunk of shared memory needs to be?
		# it is going to be determined by several factors.
		#	number of unique indices
		#	local size
		#	any ops like range
		# so it should look like:
#"""
#		local_size_flat = np.prod(local_size)
#		cache_size = 0
#		for idx_op in self._index_ops[b]:
#			cache_size += local_size_flat*get_total_indices(idx_op)
#"""
		# where local_size_flat is the flattened local size,
		# get_total_indices is a to-be-written function that gets the total number of indices an operation can provide,
		# and cache_size is the size of a given buffer's cache.
		# still need to take multiple types into account...this might be a pain in the bum
		self._index_ops[b].add(idx_op)
	
	def buf_map(self, dt:DType) -> str:
		if dt.base == dtypes.bool:
			return self.type_map[dtypes.uchar]
		return self.type_map[dt.base]
	
	def render_cast(self, dt:DType, val: str) -> str:
		if isinstance(dt, PtrDType):
			dt = dt.base
		return f"{self.type_map[dt]}({val})"
	
	def render_dtype(self, dt:DType, mutable=True) -> str:
		return f"{self.type_map[dt]}"
	
	def _render_cast_index(self, b):
		self.add_index_op(b.src[0].src[0], b.src[0])
		#self._index_ops.add(b.src[0])
		assert b.op == Ops.CAST and b.src[0].op == Ops.INDEX
		buf, idx = b.src[0].src[0], b.src[0].src[1]
		assert not buf.dtype.addrspace in [AddrSpace.LOCAL, AddrSpace.REG]
		cast_buf_str = f"push.{self[buf]} + ({self[idx]}*{buf.dtype.base.itemsize})"
		return f"{self.render_bda(b.dtype.base)}({cast_buf_str}).data[0]"
	
	def render_local_cache(self, b):
		raise NotImplementedError
	
	def render_complete_load(self, b):
		if b.op == Ops.CAST and isinstance(b.dtype, PtrDType) and b.dtype.base.count > 1:
			if not (b.dtype.addrspace in (AddrSpace.REG, AddrSpace.LOCAL) ):
				return self._render_cast_index(b)
			buf = b.src[0].src[0]
			idx = b.src[0].src[1]
			b_rendered = f"LOAD_VEC{b.dtype.base.count}({self[buf]}, {self[idx]}, {self.render_dtype(b.dtype.base)})"
		elif b.op == Ops.VECTORIZE:
			#raise ValueError("this should not happen either anymore")
			b_rendered = ""
			for idx in b.src:
				b_rendered += f"{self[idx]}, "
			b_rendered = f"{self.render_dtype(b.dtype)}({b_rendered.strip(', ')})"
		else:
			b_rendered = self[b]
			if b.dtype.base is dtypes.bool:
				# booleans in GLSL are 32-bit, but tinygrad expects them to be stored as 8-bit.
				# so they must be converted on every load (and store)
				b_rendered = f"bool({b_rendered})"
		return b_rendered
	
	def render_load(self, x:str, dt:DType) -> str:
		if dt.base == dtypes.bool:
			return f"{self.render_dtype(dtypes.bool)}({x})"
		return x
		
	def _render_macros(self):
		macros = """
struct read_cache_entry {
	uint index;
	bool stored;
};

#define zero_entry = read_cache_entry(0, false)
#define ZERO_CACHE(buf, size) \
{ \
	if (gl_LocalInvocationID.x == gl_LocalInvocationID.y == gl_LocalInvocationID.z == 0) \
	{ \
		for (uint i = 0; i < size; i += 1) buf[i] = zero_entry; \
	} \
	barrier(); \
} \

struct vec16 {
	float data[16];
};

vec16 make_vec16(float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7, float x8, float x9, float x10, float x11, float x12, float x13, float x14, float x15)
{
	return vec16(float[16](x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15) );
}

#define ZERO_VEC16 make_vec16(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

float dot(vec16 a, vec16 b)
{
	float d = 0.0;
	for (uint i = 0; i < 16; i += 4)
	{
		vec4 a4 = vec4(a.data[i], a.data[i+1], a.data[i+2], a.data[i+3]);
		vec4 b4 = vec4(b.data[i], b.data[i+1], b.data[i+2], b.data[i+3]);
		d += dot(a4, b4);
	}
	return d;
}

vec16 add(vec16 a, vec16 b)
{
	vec16 d = ZERO_VEC16;
	for (uint i = 0; i < 16; i += 4)
	{
		vec4 a4 = vec4(a.data[i], a.data[i+1], a.data[i+2], a.data[i+3]);
		vec4 b4 = vec4(b.data[i], b.data[i+1], b.data[i+2], b.data[i+3]);
		vec4 c = a4 + b4;
		d.data[i] += c[0];
		d.data[i+1] += c[1];
		d.data[i+2] += c[2];
		d.data[i+3] += c[3];
	}
	return d;
}

vec16 WMMA_4_4_1_float_float(vec4 a, vec4 b, vec16 c)
{
	return ZERO_VEC16;
}

float WMMA_1_1_1_float_float(float a, float b, float c)
{
	return fma(a, b, c);
}
#if 0
vec4 WMMA_2_2_1_float_float(vec2 a, vec2 b, vec4 c)
{
	mat2 ab = outerProduct(a);
	return c + vec4(ab[0][0], ab[0][1], ab[1][0], ab[1][1]);
}
#endif
#define ADDR_T uint64_t
#define INFINITY uintBitsToFloat(0x7F800000)
#define NAN uintBitsToFloat(0x7FC00000)
#define PI 3.14159265358979323846
#define STORE_VEC(buf, idx, vector, size, VEC_T) { VEC_T tmpV = vector; for (uint i = 0; i < size; i += 1) buf[idx + i] = tmpV[i]; }
#define LOAD_VEC2(buf, idx, VEC_T) VEC_T(buf[idx], buf[idx + 1])
#define LOAD_VEC4(buf, idx, VEC_T) VEC_T(buf[idx], buf[idx + 1], buf[idx + 2], buf[idx + 3])
precise float exp2_precise(float t) { precise float s = exp2(t); return s; }
precise float log2_precise(float t) { if (t == 0.0) return -1.0*INFINITY; precise float s = log2(t); return s; }
"""
		if self._vkdev.metadata.double: macros += """
precise float64_t exp2(float64_t x) { precise float tmp =  float(x); tmp = pow(2.0, tmp); return float64_t(tmp); }
precise float64_t exp2_precise(float64_t t) { precise float64_t s = exp2(t); return s; }
precise float64_t log2_precise(float64_t t) { if (t == 0.0) return float64_t(-1.0*INFINITY); precise float s = log2(float(t)); return float64_t(s); }
"""
		if self._vkdev.metadata.half: macros += """
precise float16_t exp2_precise(float16_t t) { precise float16_t s = exp2(t); return s; }
precise float16_t log2_precise(float16_t t) { if (t == 0.0) return float16_t(-1.0*INFINITY); precise float16_t s = log2(t); return s; }
"""
		return macros
		
	def render_bda(self, dtype):
		dt_str = self.buf_map(dtype)
		return f"{dt_str}_ptr"
		
	def _declare_buffer_references(self):
		body = ""
		added = []
		for dtype in self.type_map.keys():
			dt_str = self.buf_map(dtype)
			if dt_str in added: continue
			added.append(dt_str)
			buf_reference_type = self.render_bda(dtype)
			#body += f"layout(buffer_reference, std430, buffer_reference_align = {dtype.base.itemsize}) buffer {buf_reference_type} {{ {dt_str} data[]; }};\n"
			# TODO: Hardware may have a different optimal memory alignment size, implement a way to query this in tart
			alignment = dtype.base.itemsize if dtype.base.itemsize > 16 else 16
			body += f"layout(buffer_reference, std140, buffer_reference_align = {alignment}) buffer {buf_reference_type} {{ {dt_str} data[]; }};\n"
		return body
		
	
	def _render_extensions(self):
		extensions = {
			"#extension GL_EXT_control_flow_attributes : enable": True,
			"#extension GL_EXT_buffer_reference : require": self.supports_float4,
			"#extension GL_EXT_shader_explicit_arithmetic_types: require": True,
			"#extension GL_EXT_shader_explicit_arithmetic_types_int64: require": self._vkdev.metadata.long and self._vkdev.metadata.ulong,
			"#extension GL_EXT_shader_16bit_storage : require": self._vkdev.metadata.half or (self._vkdev.metadata.short and self._vkdev.metadata.ushort),
			"#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require": self._vkdev.metadata.double,
			"#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require": self._vkdev.metadata.half,
			"#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require": self._vkdev.metadata.short and self._vkdev.metadata.ushort,
			
			"#extension GL_EXT_shader_8bit_storage : require": self._vkdev.metadata.char and self._vkdev.metadata.uchar,
			"#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require": self._vkdev.metadata.char and self._vkdev.metadata.uchar
		}
		return "\n".join([k if k[v] else "" for k, v in extensions.items()]) + "\n"
	
	def _get_rw(self, buf_name, uops):
		read = False
		write = False
		# first thing to do is to find the uop that corresponds to this buffer...
		target_uop = None
		for u in uops:
			if u.op == Ops.DEFINE_GLOBAL:
				# this is the rendered thingy according to cstyle...
				if buf_name == f"data{u.arg}_{sz}" if (sz:=u.ptrdtype.size) > 0 else f"data{u.arg}":
					target_uop = u
					break
		if target_uop is None:
			# something else is wrong if the uop wasn't found
			raise ValueError
		use = False
		for uop in uops:
			if len(uop.src) == 0: continue
			if uop.src[0].op == Ops.INDEX:
				idx_op = uop.src[0]
				if not (idx_op.src[0] is target_uop): continue
				if uop.op == Ops.LOAD:
					read = True
					use = True
				elif uop.op == Ops.STORE:
					write = True
					use = True
		return True, read, write
		
	def render_kernel(self, function_name:str, kernel:list[str], bufs:list[tuple[str,tuple[DType,bool]]], uops:list[UOp], prefix=None) -> str:
		local_size = [u.src[0].ssimplify() for u in sorted([u for u in uops if u.op is Ops.SPECIAL and u.arg[0] == 'l'], key=lambda u: u.arg)]
		if not local_size: local_size = [1]
		while len(local_size) < 3: local_size.append(1)
		bind_it = iter(range(len(bufs)))
		external_local_bufs = [line.lstrip() for line in kernel if "shared " in line]
		kernel[:] = [line for line in kernel if "shared " not in line]
		prg = ""
		prg += "\nfloat nan() { uint bits = 0xffffffffu; return uintBitsToFloat(bits); }\n"
		lst = []
		push_const_lst = []
		
		for name, (dtype, buf) in bufs:
			if isinstance(dtype, PtrDType):
				if self.supports_float4:
					push_const_lst.append(f"	ADDR_T {name};")
				else:
					_, read, write = self._get_rw(name, uops)
					rw_str = ""
					if read and (not write): rw_str = "readonly"
					elif write and (not read): rw_str = "writeonly"
					buf_str = f"layout(set = 0, binding = {next(bind_it)}"
					buf_str += f", std430) {rw_str} buffer {name}_buf {{ {self.buf_map(dtype.base)} {name}[]; }};"
					lst.append(buf_str)
			else:
				push_const_lst.append(f"	{self.buf_map(dtype.base)} {name};")
		if len(push_const_lst) > 0: lst.append(f"layout(push_constant) uniform push_consts {{ {' '.join(push_const_lst)} }} push;")
		prg += ("\n".join((external_local_bufs or []) + lst) + f"\nvoid main()\n")
		header = "#version 450\n" + self._render_extensions() + "\n"
		header += f"layout(local_size_x = {local_size[0]}, local_size_y = {local_size[1]}, local_size_z = {local_size[2]}) in;\n"
		header += self._render_macros()
		if self.supports_float4: header += self._declare_buffer_references()
		
		# add the ZERO_CACHE operations
		
		out = header + prg + "{\n" + "\n".join(kernel) + "\n}"
		if SAVE_RENDERED_KERNELS:
			self._path = "rendered"
			os.makedirs(self._path, exist_ok = True)
			with open(os.path.join(self._path, f"{function_name}.glsl"), "w") as f:
				f.write(out)
		self._index_ops.clear()
		return out
