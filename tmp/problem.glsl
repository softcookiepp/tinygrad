#version 450
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_ARB_gpu_shader_int64: require
#extension GL_ARB_gpu_shader_int64: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_ARB_compute_shader: require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
#define INFINITY uintBitsToFloat(0x7F800000)

float nan() { uint bits = 0xffffffffu; return uintBitsToFloat(bits); }
layout(set = 0, binding = 0, std430) buffer data0_1_buf { uint8_t data0_1[]; };
layout(set = 0, binding = 1, std430) buffer data1_2_buf { uint8_t data1_2[]; };
void main()
{
  bool val0 = bool(data1_2[0]);
  bool val1 = bool(data1_2[1]);
  data0_1[0] = uint8_t(((val0!=true) || (val1!=true)!=true));
}

