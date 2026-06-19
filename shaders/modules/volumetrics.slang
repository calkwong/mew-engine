module volumetrics;

import bindings;
import util;

internal static const float PI = 3.14159265359;

public float phase_schlick(float g, float cos_theta)
{
    float numerator = 1.0 - g * g;
    float g_cos_theta = g * cos_theta;
    float denominator = 4.0 * PI * ((1.0 + g_cos_theta) * (1.0 + g_cos_theta));
    return numerator / denominator;
}

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
internal float generate_noise(int2 frag_coord, float frame, float noise_scale, uint blue_noise_texture_id)
{
    RWTexture2D blue_noise_tex = get_storage_image<RWTexture2D>(blue_noise_texture_id);
    float2 blue_noise = blue_noise_tex.Load(frag_coord).rg;

    const float k_golden_ratio_conjugate = 0.61803398875;
    float blue_noise0 = frac(blue_noise.r + float(frame % 256) * k_golden_ratio_conjugate);
    float blue_noise1 = frac(blue_noise.g + float(frame % 256) * k_golden_ratio_conjugate);

    return triangular_noise(blue_noise0, blue_noise1) * noise_scale;
}

public float3 jittered_world_from_froxel(uint3 coord, uint3 froxel_dim, float4x4 inverse_view_proj, float near, float far, uint frame)
{
    float2 uv = (float2(coord.xy) + float2(0.5)) / float2(froxel_dim.xy);
    float froxel_z = float(froxel_dim.z);
    float slice = float(coord.z);

    float jitter_z = get_ign_jitter(frame);
    // float jitter_z = 0.0;

    float linear_depth = near * pow(far / near, (slice + 0.5 + jitter_z) / froxel_z); // no negation as we want +ve linear depth
    float depth = linear_depth_to_raw_depth(linear_depth, near);

    return reconstruct_world_pos(depth, inverse_view_proj, uv);
}

public float3 world_from_froxel(uint3 coord, uint3 froxel_dim, float4x4 inverse_view_proj, float near, float far)
{
    float2 uv = (float2(coord.xy) + float2(0.5)) / float2(froxel_dim.xy);
    float froxel_z = float(froxel_dim.z);
    float slice = float(coord.z);

    float linear_depth = near * pow(far / near, (slice + 0.5) / froxel_z); // no negation as we want +ve linear depth
    float depth = linear_depth_to_raw_depth(linear_depth, near);

    return reconstruct_world_pos(depth, inverse_view_proj, uv);
}

public float view_z_from_slice(uint slice, uint num_slices, float near, float far)
{
    return near * pow(far / near, (slice + 0.5) / num_slices);
}

// https://www.mattiasstrand.com/posts/volumetric-fog/
internal float IGN(int pixel_x, int pixel_y)
{
    return fmod(52.9829189f * fmod(0.06711056f * float(pixel_x) + 0.00583715f * float(pixel_y), 1.0f), 1.0f);
}

public float get_ign_jitter(uint frame_index)
{
    return IGN(frame_index & 1023, 0);
}
