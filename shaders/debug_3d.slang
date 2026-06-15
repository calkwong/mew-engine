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
    RWTexture2D out_tex = get_storage_image<RWTexture2D>(pc.draw_id);
    RWTexture3D debug_tex = get_storage_image<RWTexture3D>(pc.debug_texture_id);

    uint layers;

    uint3 debug_resolution;
    debug_tex.GetDimensions(0, debug_resolution.x, debug_resolution.y, debug_resolution.z, layers);

    uint x = global_id.x;
    uint y = global_id.y;

    const uint tile_size = 5;

    if (any(uint2(x, y) >= debug_resolution.xy * tile_size))
        return;

    uint2 uv = uint2(float2(x, y) / float(tile_size));

    float4 output = debug_tex.Load(uint3(uv, pc.slice));
    out_tex.Store(uint2(x, y), float4(output.xyz, 1));
}
