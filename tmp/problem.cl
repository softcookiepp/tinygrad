__kernel void r_2(__global bool* data0_1, __global bool* data1_2) {
  bool val0 = (*(data1_2+0));
  bool val1 = (*(data1_2+1));
  *(data0_1+0) = (((val0!=1)|(val1!=1))!=1);
}
