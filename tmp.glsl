#version 450
#ifndef VULKAN
#define VULKAN 100
#endif


layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer buf_data0 { float16_t data0[]; };
layout(set = 0, binding = 1) buffer buf_data1 { float data1[]; };
void E_128_32_4n2() {
  int gidx0 = int( gl_WorkGroupID[0] ); /* 128 */
  int lidx0 = int( gl_LocalInvocationID[0] ); /* 32 */
  int alu0 = ((gidx0<<7)+(lidx0<<2));
  float val0 = data1[alu0];
  int alu1 = (alu0+1);
  float val1 = data1[alu1];
  int alu2 = (alu0+2);
  float val2 = data1[alu2];
  int alu3 = (alu0+3);
  float val3 = data1[alu3];
  data0[alu0] = (float16_t(val0));
  data0[alu1] = (float16_t(val1));
  data0[alu2] = (float16_t(val2));
  data0[alu3] = (float16_t(val3));
}
void main() { E_128_32_4n2(); }