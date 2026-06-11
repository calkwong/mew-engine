import bindings;
import util;

// needs tweaking to work

struct PushConstants
{
    uint2 swapchain_resolution;
    uint debug_texture_id;
    uint draw_id;
    uint slice;
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(32, 32, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    RWTexture3D debug_tex = get_storage_image<RWTexture3D>(pc.debug_texture_id);
    RWTexture2D out_tex = get_storage_image<RWTexture2D>(pc.draw_id);

    uint layers;
    uint3 debug_resolution;
    debug_tex.GetDimensions(0, debug_resolution.x, debug_resolution.y, debug_resolution.z, layers);

    uint x = global_id.x;
    uint y = global_id.y;

    if (x >= debug_resolution.x || y >= debug_resolution.y)
        return;

    float4 output = debug_tex.Load(uint3(x, y, pc.slice));
    out_tex.Store(uint2(x, y), output);

    // float output = debug_tex.Load(uint3(x, y, pc.slice)).g;
    // out_tex.Store(uint2(x, y), float4(float3(output), 1));
}
