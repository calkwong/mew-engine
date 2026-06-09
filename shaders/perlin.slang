import bindings;
import util;

struct PushConstants
{
    uint2 resolution;
    uint render_target;
}

[[vk::push_constant]]
PushConstants pc;

static const float PI = 3.14159265359;

float fade(float t)
{
    return t * t * t * (t * (t * 6 - 15) + 10);
}

float2 random_gradient_vector(uint seed)
{
    float phi = 2.0 * PI * random_value(seed);

    return float2(cos(phi), sin(phi));
}

// https://www.shadertoy.com/view/XlGcRh
// SuperFastHash, adapated from http://www.azillionmonkeys.com/qed/hash.html
uint superfast(uint2 data)
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

    if (x >= pc.resolution.x || y >= pc.resolution.y)
        return;

    const float frequency = 64.0;
    float2 sample = (float2(x, y) + float2(0.5)) / float2(pc.resolution) * frequency;

    uint sx = uint(floor(sample.x));
    uint sy = uint(floor(sample.y));

    uint seed0 = superfast(uint2(sx, sy) + uint2(0, 0));
    uint seed1 = superfast(uint2(sx, sy) + uint2(1, 0));
    uint seed2 = superfast(uint2(sx, sy) + uint2(0, 1));
    uint seed3 = superfast(uint2(sx, sy) + uint2(1, 1));

    float2 g0 = random_gradient_vector(seed0);
    float2 g1 = random_gradient_vector(seed1);
    float2 g2 = random_gradient_vector(seed2);
    float2 g3 = random_gradient_vector(seed3);

    float2 s0 = sample - (float2(sx, sy) + float2(0, 0));
    float2 s1 = sample - (float2(sx, sy) + float2(1, 0));
    float2 s2 = sample - (float2(sx, sy) + float2(0, 1));
    float2 s3 = sample - (float2(sx, sy) + float2(1, 1));

    float d0 = dot(g0, s0);
    float d1 = dot(g1, s1);
    float d2 = dot(g2, s2);
    float d3 = dot(g3, s3);

    // interpolation weights
    float ix = frac(sample.x);
    float iy = frac(sample.y);

    ix = fade(ix);
    iy = fade(iy);

    float noise_value = lerp(
        lerp(d0, d1, ix),
        lerp(d2, d3, ix),
        iy
    );
    noise_value = noise_value * 0.5 + 0.5;

    RWTexture2D tex = get_sampled_image<RWTexture2D>(pc.render_target);
    tex.Store(uint2(x, y), float4(float3(noise_value), 1.0));
}
