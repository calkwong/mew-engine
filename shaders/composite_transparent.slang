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
        RWTexture2D tex = get_storage_image<RWTexture2D>(pc.draw_id);
        float3 color = tex.Load(int2(texel_coord)).rgb;

        uint index = texel_coord.x + texel_coord.y * pc.screen_size.x;
        bool update = false;
        color = composite_transparent(pc.oit_buffer, index, color, texel_coord, update);

        if (update)
            tex.Store(texel_coord, float4(color, 1.0));
    }
}
