import bindings;
import util;

struct PushConstants
{
    uint2 resolution; // noise tex resolution
    uint noise_id;
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
    uint x = global_id.x;
    uint y = global_id.y;

    if (x >= pc.resolution.x || y >= pc.resolution.y)
        return;

    RWTexture3D<float> noise_tex = get_storage_image<RWTexture3D<float>>(pc.noise_id);
    RWTexture2D out_tex = get_storage_image<RWTexture2D>(pc.draw_id);

    float noise = noise_tex.Load(uint3(x, y, pc.slice)).r;

    // note: no safety check, we assume draw_image x/y dimensions >= noise tex
    out_tex.Store(uint2(x, y), float4(float3(noise), 1));
}
