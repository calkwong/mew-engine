import bindings;
import types;
import prefix_sum;

struct PushConstants
{
    PrefixSum* prefix_sum_buffer;
    Dispatch* dispatch_buffer;
};

[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    uint count = uint(get_prefix_sum_counter(pc.prefix_sum_buffer));

    // TODO: upload to pc else if # of threads change in culling shader, this breaks
    const uint threads = 256;

    pc.dispatch_buffer.workgroup_x = (count + threads - 1) / threads;
    pc.dispatch_buffer.workgroup_y = 1;
    pc.dispatch_buffer.workgroup_z = 1;
}
