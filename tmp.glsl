#version 450
#ifndef VULKAN
#define VULKAN 100
#endif


layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer buf_data0 { float data0[]; };
layout(set = 0, binding = 1) buffer buf_data1 { float data1[]; };
layout(set = 0, binding = 2) buffer buf_data2 { float data2[]; };
void E_1848_32_2() {
  int gidx0 = int( gl_WorkGroupID[0] ); /* 1848 */
  int lidx0 = int( gl_LocalInvocationID[0] ); /* 32 */
  int alu0 = (lidx0+(gidx0<<5));
  float val0 = data1[alu0];
  float val1 = data2[alu0];
  data0[alu0] = (float((float16_t(val0))));
  data0[(alu0+59136)] = (float((float16_t(val1))));
}
void main() { E_1848_32_2(); }