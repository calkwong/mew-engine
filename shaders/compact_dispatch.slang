#version 460

#extension GL_GOOGLE_include_directive : require

#include "bindings.glsl"
#include "buffer_references.glsl"

#include "prefix_sum.glsl"

layout (local_size_x = 1) in;

layout(push_constant) uniform constants
{
    PrefixSumBuffer prefix_sum_buffer;
    DispatchBuffer dispatch_buffer;
};

// TODO this should be between mesh cull and meshlet cull

void main()
{
    uint count = uint(get_prefix_sum_counter(prefix_sum_buffer));

    const uint threads = 256;

    dispatch_buffer.workgroup_x = (count + threads - 1) / threads;
    dispatch_buffer.workgroup_y = 1;
    dispatch_buffer.workgroup_z = 1;
}
