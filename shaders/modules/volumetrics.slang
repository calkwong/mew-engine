module volumetrics;

import bindings;
import util;

internal static const float PI = 3.14159265359;

public uint3 depth_to_froxel(float depth, float near, float volumetrics_scale, float volumetrics_bias, float2 frag_coord, float2 screen_dim, uint3 froxel_dims)
{
    float linear_depth = near / depth; // infinite plane + reverse z
    uint slice = uint(floor(log(linear_depth) * volumetrics_scale - volumetrics_bias));
    slice = clamp(slice, 0, froxel_dims.z - 1);

    float2 pixels_per_froxel = screen_dim / float2(froxel_dims.xy);
    uint2 froxel_index = uint2(floor(frag_coord / pixels_per_froxel));

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

// Takes 2 noises in space [0, 1] and remaps to [-1, 1]
internal float triangular_noise(float noise0, float noise1)
{
    return noise0 + noise1 - 1.0f;
}

// Animated blue noise using golden ratio
internal float generate_noise(float2 uv, float frame, float noise_scale, uint blue_noise_texture_id)
{
    Texture2D blue_noise_tex = get_sampled_image<Texture2D>(blue_noise_texture_id);
    SamplerState linear_samp = get_sampler<SamplerState>(LINEAR_SAMPLER);
    float2 blue_noise = blue_noise_tex.SampleLevel(linear_samp, uv, 0.0).rg;

    const float k_golden_ratio_conjugate = 0.61803398875;
    float blue_noise0 = frac(blue_noise.r + float(frame % 256) * k_golden_ratio_conjugate);
    float blue_noise1 = frac(blue_noise.g + float(frame % 256) * k_golden_ratio_conjugate);

    return triangular_noise(blue_noise0, blue_noise1) * noise_scale;
}

public float3 world_from_froxel(uint3 coord, uint3 froxel_dim, float4x4 inverse_view_proj, float near, float far, float2 halton, float temporal_reprojection_jitter_scale, uint blue_noise_texture_id, uint frame)
{
    // float2 uv = (float2(coord.xy) + float2(0.5) + halton * temporal_reprojection_jitter_scale) / float2(froxel_dim.xy);
    float2 uv = (float2(coord.xy) + float2(0.5)) / float2(froxel_dim.xy);
    float froxel_z = float(froxel_dim.z);
    float slice = float(coord.z);

    float jitter = 0.0;
    // float noise_scale = 1.0; // TODO: make this a cvar
    // float jitter = generate_noise(float2(coord.xy) / float2(froxel_dim.xy), frame, noise_scale, blue_noise_texture_id);

    // from ID tech
    float linear_depth = near * pow(far / near, (slice + 0.5 + jitter) / froxel_z); // no negation as we want +ve linear depth
    // float depth = linear_depth_to_raw_depth(linear_depth, pc.far, pc.near); // reverse-z
    float depth = linear_depth_to_raw_depth(linear_depth, near); // infinite far plane + reverse z

    return reconstruct_world_pos(depth, inverse_view_proj, uv);
}

public float3 world_from_froxel_no_jitter(uint3 coord, uint3 froxel_dim, float4x4 inverse_view_proj, float near, float far)
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
