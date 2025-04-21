#version 450
#ifndef VULKAN
#define VULKAN 100
#endif


layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(set = 0, binding = 0) buffer buf_data0 { float16_t data0[]; };
layout(set = 0, binding = 1) buffer buf_data1 { float data1[]; };
layout(set = 0, binding = 2) buffer buf_data2 { int data2[]; };
void r_5_2_10n2() {
  int gidx0 = int( gl_WorkGroupID[0] ); /* 5 */
  int lidx0 = int( gl_LocalInvocationID[0] ); /* 2 */
  int alu0 = (lidx0+(gidx0<<1));
  int val0 = data2[alu0];
  float val1 = data1[alu0];
  float val2 = data1[(alu0+10)];
  data0[alu0] = (sin(((float16_t((((alu0<9)?0.0f:1.0f)+((gidx0<4)?0.0f:1.0f)+((alu0<7)?0.0f:1.0f)+((gidx0<3)?0.0f:1.0f)+((alu0<5)?0.0f:1.0f)+((gidx0<2)?0.0f:1.0f)+((alu0<3)?0.0f:1.0f)+((gidx0<1)?0.0f:1.0f)+((alu0<1)?0.0f:1.0f)+1.0f)))+(float16_t((sin((1.5707963267948966f+(val1*-6.283185307179586f)))*sqrt((log2((1.0f-val2))*-1.3862943611198906f)))))+(float16_t(-1.0f))))*(float16_t(1)/((float16_t(((float((val0+-1)))*0.5555555555555556f)))+(float16_t(0.01f))))*(float16_t(20.0f)));
}
void main() { r_5_2_10n2(); }