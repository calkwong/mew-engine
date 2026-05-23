import bindings;
import oit;

struct PushConstants
{
    OITData* oit_buffer;
    uint2 screen_size;
    uint draw_id;
}

[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    uint2 texel_coord = global_id.xy;

    if (texel_coord.x < pc.screen_size.x && texel_coord.y < pc.screen_size.y)
    {
        float3 color = images_rgba32f[pc.draw_id].Load(int2(texel_coord)).rgb;

        uint index = texel_coord.x + texel_coord.y * pc.screen_size.x;
        color = composite_transparent(pc.oit_buffer, index, color, texel_coord);

        images_rgba32f[pc.draw_id].Store(texel_coord, float4(color, 1.0));
    }
}
