import bindings;

struct PushConstants
{
    uint3 froxel_dims;
    uint light_scattering_tex;
    uint scattering_extinction_tex; // this is a texture reuse
};

[[vk::push_constant]]
PushConstants pc;

#define SIGMA_FILTER 4.0
#define RADIUS 2

float gaussian(float radius, float sigma)
{
    float v = radius / sigma;
    return exp(-(v * v));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    uint3 froxel_coord = global_id;

    if (any(froxel_coord >= pc.froxel_dims))
        return;

    // TODO: use proper sampler
    SamplerState samp = get_sampler<SamplerState>(NEAREST_SAMPLER);
    Texture3D light_scattering_tex = get_sampled_image<Texture3D>(pc.light_scattering_tex);

    float3 rcp_froxel_dims = 1.0 / float3(pc.froxel_dims);

    float accumulated_weight = 0.0;
    float4 accumulated_scattering_extinction = float4(0.0);

    for (int i = -RADIUS; i <= RADIUS; i++)
    {
        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            int3 coord = int3(froxel_coord) + int3(i, j, 0);
            if (all(coord >= 0) && all(coord < int3(pc.froxel_dims)))
            {
                float weight = gaussian(length(int2(i, j)), SIGMA_FILTER);
                float3 sample_coord = (float3(coord) + float3(0.5)) * rcp_froxel_dims;
                float4 sampled_value = light_scattering_tex.SampleLevel(samp, sample_coord, 0.0);
                accumulated_scattering_extinction += sampled_value * weight;
                accumulated_weight += weight;
            }
        }
    }

    float4 scattering_extinction = accumulated_scattering_extinction / accumulated_weight;

    RWTexture3D scattering_extinction_tex = get_storage_image<RWTexture3D>(pc.scattering_extinction_tex);
    scattering_extinction_tex.Store(froxel_coord, scattering_extinction);
}
