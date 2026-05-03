module sampling;

import util;

static const float PI = 3.14159265359;

public float RadicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

public float2 Hammersley(uint i, uint N)
{
    return float2(float(i) / float(N), RadicalInverse_VdC(i));
}

public float3 ImportanceSampleGGX(float2 Xi, float3 N, float linear_roughness)
{
    float a = linear_roughness;

    float phi = 2.0 * PI * Xi.x;
    float cos_theta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sin_theta = sqrt(1.0 - cos_theta*cos_theta);

    // from spherical coordinates to cartesian coordinates
    float3 H;
    H.x = cos(phi) * sin_theta;
    H.y = sin(phi) * sin_theta;
    H.z = cos_theta;

    float3 sample_dir = tangent_to_world_space(H, N);

    return sample_dir;
}

public float3 random_cosine_hemisphere(inout uint seed, float3 N)
{
    float2 sample = float2(random_value(seed), random_value(seed));

    float phi = 2.0 * PI * sample.x;
    float x = cos(phi) * sqrt(sample.y);
    float y = sin(phi) * sqrt(sample.y);
    float z = sqrt(1.0 - sample.y);

    // local/tangent space
    return normalize(float3(x, y, z));
}

// TODO: sample VNDF function
// TODO: power/balance heuristic
