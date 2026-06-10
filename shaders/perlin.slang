import bindings;
import util;

struct PushConstants
{
    uint3 resolution; // noise tex resolution
    uint noise_id;
}

[[vk::push_constant]]
PushConstants pc;

static const float PI = 3.14159265359;

float fade(float t)
{
    return t * t * t * (t * (t * 6 - 15) + 10);
}

float3 random_gradient_vector(uint seed)
{
    float phi = 2.0 * PI * random_value(seed);
    float z = 2.0 * random_value(seed) - 1.0;
    float r = sqrt(1.0 - z * z);

    float x = cos(phi) * r;
    float y = sin(phi) * r;

    return float3(x, y, z);
}

// https://www.shadertoy.com/view/XlGcRh
// SuperFastHash, adapated from http://www.azillionmonkeys.com/qed/hash.html
uint superfast(uint3 data)
{
    uint hash = 8u, tmp;

    hash += data.x & 0xffffu;
    tmp = (((data.x >> 16) & 0xffffu) << 11) ^ hash;
    hash = (hash << 16) ^ tmp;
    hash += hash >> 11;

    hash += data.y & 0xffffu;
    tmp = (((data.y >> 16) & 0xffffu) << 11) ^ hash;
    hash = (hash << 16) ^ tmp;
    hash += hash >> 11;

    hash += data.z & 0xffffu;
    tmp = (((data.z >> 16) & 0xffffu) << 11) ^ hash;
    hash = (hash << 16) ^ tmp;
    hash += hash >> 11;

    /* Force "avalanching" of final 127 bits */
    hash ^= hash << 3;
    hash += hash >> 5;
    hash ^= hash << 4;
    hash += hash >> 17;
    hash ^= hash << 25;
    hash += hash >> 6;

    return hash;
}

[shader("compute")]
[numthreads(32, 32, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    uint x = global_id.x;
    uint y = global_id.y;
    uint z = global_id.z;

    if (x >= pc.resolution.x || y >= pc.resolution.y || z >= pc.resolution.z)
        return;

    const float frequency = 16.0;
    float3 sample = (float3(global_id) + float3(0.5)) / float3(pc.resolution) * frequency;

    uint sx = uint(floor(sample.x));
    uint sy = uint(floor(sample.y));
    uint sz = uint(floor(sample.z));

    uint seed0 = superfast(uint3(sx, sy, sz) + uint3(0, 0, 0));
    uint seed1 = superfast(uint3(sx, sy, sz) + uint3(1, 0, 0));
    uint seed2 = superfast(uint3(sx, sy, sz) + uint3(0, 1, 0));
    uint seed3 = superfast(uint3(sx, sy, sz) + uint3(1, 1, 0));
    uint seed4 = superfast(uint3(sx, sy, sz) + uint3(0, 0, 1));
    uint seed5 = superfast(uint3(sx, sy, sz) + uint3(1, 0, 1));
    uint seed6 = superfast(uint3(sx, sy, sz) + uint3(0, 1, 1));
    uint seed7 = superfast(uint3(sx, sy, sz) + uint3(1, 1, 1));

    float3 g0 = random_gradient_vector(seed0);
    float3 g1 = random_gradient_vector(seed1);
    float3 g2 = random_gradient_vector(seed2);
    float3 g3 = random_gradient_vector(seed3);
    float3 g4 = random_gradient_vector(seed4);
    float3 g5 = random_gradient_vector(seed5);
    float3 g6 = random_gradient_vector(seed6);
    float3 g7 = random_gradient_vector(seed7);

    float3 s0 = sample - (float3(sx, sy, sz) + float3(0, 0, 0));
    float3 s1 = sample - (float3(sx, sy, sz) + float3(1, 0, 0));
    float3 s2 = sample - (float3(sx, sy, sz) + float3(0, 1, 0));
    float3 s3 = sample - (float3(sx, sy, sz) + float3(1, 1, 0));
    float3 s4 = sample - (float3(sx, sy, sz) + float3(0, 0, 1));
    float3 s5 = sample - (float3(sx, sy, sz) + float3(1, 0, 1));
    float3 s6 = sample - (float3(sx, sy, sz) + float3(0, 1, 1));
    float3 s7 = sample - (float3(sx, sy, sz) + float3(1, 1, 1));

    float d0 = dot(g0, s0);
    float d1 = dot(g1, s1);
    float d2 = dot(g2, s2);
    float d3 = dot(g3, s3);
    float d4 = dot(g4, s4);
    float d5 = dot(g5, s5);
    float d6 = dot(g6, s6);
    float d7 = dot(g7, s7);

    // interpolation weights
    float ix = frac(sample.x);
    float iy = frac(sample.y);
    float iz = frac(sample.z);

    ix = fade(ix);
    iy = fade(iy);
    iz = fade(iz);

    float noise_value = lerp(
        lerp(lerp(d0, d1, ix), lerp(d2, d3, ix), iy),
        lerp(lerp(d4, d5, ix), lerp(d6, d7, ix), iy),
        iz
    );
    noise_value = noise_value * 0.5 + 0.5;

    RWTexture3D<float> tex = get_storage_image<RWTexture3D<float>>(pc.noise_id);
    tex.Store(uint3(x, y, z), noise_value);
}
