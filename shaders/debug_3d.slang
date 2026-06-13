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
    // RWTexture3D debug_tex = get_storage_image<RWTexture3D>(pc.debug_texture_id);

    uint layers;

    // uint3 debug_resolution;
    // debug_tex.GetDimensions(0, debug_resolution.x, debug_resolution.y, debug_resolution.z, layers);

    RWTexture2D debug_tex = get_storage_image<RWTexture2D>(pc.debug_texture_id);
    SamplerState samp = get_sampler<SamplerState>(NEAREST_REPEAT_SAMPLER);

    uint2 debug_resolution;
    debug_tex.GetDimensions(0, debug_resolution.x, debug_resolution.y, layers);

    uint x = global_id.x;
    uint y = global_id.y;

    if (x >= debug_resolution.x || y >= debug_resolution.y)
        return;

    // float4 output = debug_tex.Load(uint3(x, y, pc.slice));
    // float4 output = debug_tex.Load(uint2(x, y));
    // float2 output = debug_tex.SampleLevel(samp, (float2(x, y) + float2(0.5)) / float2(debug_resolution), 0.0).rg;
    float2 output = debug_tex.Load(uint2(x, y)).rg;
    out_tex.Store(uint2(x, y), float4(output.xy, 0.0, 1.0));

    // float output = debug_tex.Load(uint3(x, y, pc.slice)).g;
    // out_tex.Store(uint2(x, y), float4(float3(output), 1));
}
