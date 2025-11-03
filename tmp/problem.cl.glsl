#version 450
#if defined(GL_EXT_shader_explicit_arithmetic_types_int8)
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#elif defined(GL_NV_gpu_shader5)
#extension GL_NV_gpu_shader5 : require
#else
#error No extension available for Int8.
#endif

#ifndef SPIRV_CROSS_CONSTANT_ID_0
#define SPIRV_CROSS_CONSTANT_ID_0 1u
#endif
#ifndef SPIRV_CROSS_CONSTANT_ID_1
#define SPIRV_CROSS_CONSTANT_ID_1 1u
#endif
#ifndef SPIRV_CROSS_CONSTANT_ID_2
#define SPIRV_CROSS_CONSTANT_ID_2 1u
#endif

layout(local_size_x = SPIRV_CROSS_CONSTANT_ID_0, local_size_y = SPIRV_CROSS_CONSTANT_ID_1, local_size_z = SPIRV_CROSS_CONSTANT_ID_2) in;

layout(binding = 0, std430) buffer _11_13
{
    uint8_t _m0[];
} _13;

layout(binding = 1, std430) buffer _11_14
{
    uint8_t _m0[];
} _14;

uvec3 _8 = gl_WorkGroupSize;

void main()
{
    _13._m0[0u] = (_14._m0[0u] & uint8_t(1)) & _14._m0[1u];
}

