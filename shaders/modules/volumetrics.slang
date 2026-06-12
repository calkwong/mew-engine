module volumetrics;

import util;

internal static const float PI = 3.14159265359;

public uint3 depth_to_froxel(float depth, float near, float volumetrics_scale, float volumetrics_bias, float2 frag_coord, float2 screen_dim, uint3 froxel_dims)
{
    float linear_depth = near / depth; // infinite plane + reverse z
    uint slice = uint(floor(log(linear_depth) * volumetrics_scale - volumetrics_bias));
    slice = clamp(slice, 0, froxel_dims.z - 1);

    float2 ratio = float2(froxel_dims.xy) / screen_dim;
    uint2 froxel_index = uint2(floor(ratio * frag_coord)); // implicit floor?
    froxel_index.x = clamp(froxel_index.x, 0, froxel_dims.x - 1);
    froxel_index.y = clamp(froxel_index.y, 0, froxel_dims.y - 1);

    return uint3(froxel_index.x, froxel_index.y, slice);
}

public float phase_shlick(float g, float cos_theta)
{
    float numerator = 1.0 - g * g;
    float g_cos_theta = g * cos_theta;
    float denominator = 4.0 * PI * ((1.0 + g_cos_theta) * (1.0 + g_cos_theta));
    return numerator / denominator;
}

// public float linear_depth_to_raw_depth(float linear_depth, float near, float far)
// {
//     return (near * far) / (linear_depth * (near - far)) - far / (near - far);
// }

// infinite far plane + reverse-z
public float linear_depth_to_raw_depth(float linear_depth, float near)
{
    return near / linear_depth;
}

public float3 world_from_froxel(uint3 coord, uint3 froxel_dim, float4x4 inverse_view_proj, float near, float far)
{
    float2 uv = (float2(coord.xy) + float2(0.5)) / float2(froxel_dim.xy);
    float froxel_z = float(froxel_dim.z);
    float slice = float(coord.z);

    // from ID tech
    float linear_depth = near * pow(far / near, (slice + 0.5) / froxel_z); // no negation as we want +ve linear depth
    // float depth = linear_depth_to_raw_depth(linear_depth, pc.far, pc.near); // reverse-z
    float depth = linear_depth_to_raw_depth(linear_depth, near); // infinite far plane + reverse z

    return reconstruct_world_pos(depth, inverse_view_proj, uv);
}
