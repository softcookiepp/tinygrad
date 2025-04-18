#version 450
#ifndef VULKAN
#define VULKAN 100
#endif


layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer buf_data0 { float data0[]; };
layout(set = 0, binding = 1) buffer buf_data1 { float data1[]; };
void r_5_2_10n1() {
  int gidx0 = int( gl_WorkGroupID[0] ); /* 5 */
  int lidx0 = int( gl_LocalInvocationID[0] ); /* 2 */
  int alu0 = (lidx0+(gidx0<<1));
  float val0 = data1[alu0];
  float val1 = data1[(alu0+10)];
  float cast0 = (float((((gidx0<4)?0:1)+((alu0<9)?0:1)+((alu0<7)?0:1)+((gidx0<3)?0:1)+((alu0<5)?0:1)+((gidx0<2)?0:1)+((alu0<3)?0:1)+((gidx0<1)?0:1)+((alu0<1)?0:1))));
  data0[alu0] = (sin((cast0+(sin((1.5707963267948966f+(val0*-6.283185307179586f)))*sqrt((log2((1.0f-val1))*-1.3862943611198906f)))))*(1/((cast0*0.5555555555555556f)+0.01f))*20.0f);
}
void main() { r_5_2_10n1(); }