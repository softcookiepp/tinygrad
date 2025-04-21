#version 450
#ifndef VULKAN
#define VULKAN 100
#endif


layout(local_size_x = 2, local_size_y = 3, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer buf_data0 { float data0[]; };
layout(set = 0, binding = 1) buffer buf_data1 { float data1[]; };
layout(set = 0, binding = 2) buffer buf_data2 { float data2[]; };
layout(set = 0, binding = 3) buffer buf_data3 { float data3[]; };
void r_5_251_29_2_3_3_6_3_6() {
  int gidx0 = int( gl_WorkGroupID[0] ); /* 29 */
  int gidx1 = int( gl_WorkGroupID[1] ); /* 251 */
  int gidx2 = int( gl_WorkGroupID[2] ); /* 5 */
  int lidx0 = int( gl_LocalInvocationID[0] ); /* 2 */
  int lidx1 = int( gl_LocalInvocationID[1] ); /* 3 */
  int alu0 = (lidx1*3);
  int alu1 = (gidx0*9);
  float acc0 = 0.0;
  float acc1 = 0.0;
  float acc2 = 0.0;
  for (int ridx5 = 0; ridx5 < 3; ridx5++) {
    for (int ridx6 = 0; ridx6 < 6; ridx6++) {
      int alu2 = ((ridx6*6)+(gidx2*108)+(ridx5*36));
      float val0 = data2[alu2];
      int alu3 = ((ridx6*266)+(ridx5*68096)+alu0+(lidx0*204288)+alu1+(gidx1*266));
      float val1 = data1[alu3];
      float val2 = data2[(alu2+1)];
      float val3 = data2[(alu2+2)];
      float val4 = data2[(alu2+3)];
      float val5 = data2[(alu2+4)];
      float val6 = data2[(alu2+5)];
      float val7 = data1[(alu3+1)];
      float val8 = data1[(alu3+2)];
      float val9 = data1[(alu3+3)];
      float val10 = data1[(alu3+4)];
      float val11 = data1[(alu3+5)];
      float val12 = data1[(alu3+6)];
      float val13 = data1[(alu3+7)];
      acc0 = (acc0+(val1*val0)+(val7*val2)+(val8*val3)+(val9*val4)+(val10*val5)+(val11*val6));
      acc1 = (acc1+(val7*val0)+(val8*val2)+(val9*val3)+(val10*val4)+(val11*val5)+(val12*val6));
      acc2 = (acc2+(val8*val0)+(val9*val2)+(val10*val3)+(val11*val4)+(val12*val5)+(val13*val6));
    }
  }
  float val14 = data3[gidx2];
  int alu9 = (alu0+(lidx0*327555)+alu1+(gidx1*261)+(gidx2*65511));
  data0[alu9] = (acc0+val14);
  data0[(alu9+1)] = (acc1+val14);
  data0[(alu9+2)] = (acc2+val14);
}
void main() { r_5_251_29_2_3_3_6_3_6(); }