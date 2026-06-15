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
    Texture3D<float4> debug_tex = get_sampled_image<Texture3D<float4>>(pc.debug_texture_id);
    SamplerState samp = get_sampler<SamplerState>(LINEAR_CLAMP_SAMPLER);

    uint layers;

    uint3 debug_resolution;
    debug_tex.GetDimensions(0, debug_resolution.x, debug_resolution.y, debug_resolution.z, layers);

    uint x = global_id.x;
    uint y = global_id.y;

    const float2 tile_size = float2(pc.swapchain_resolution) / float2(debug_resolution.xy);

    if (any(uint2(x, y) >= pc.swapchain_resolution))
        return;

    uint2 uv = uint2(float2(x, y) / tile_size);

    float4 output = debug_tex.SampleLevel(samp, float3(uv, pc.slice) / float3(debug_resolution), 0.0);
    out_tex.Store(uint2(x, y), float4(output.xyz, 1));
}
