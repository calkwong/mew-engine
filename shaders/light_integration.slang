import bindings;

struct PushConstants
{
    float4x4 inverse_view_proj;
    uint3 froxel_dimensions;
    float near;
    float far;
    uint light_scattering_tex;
    uint integrated_light_scattering_tex;
};

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    uint3 froxel_coord = global_id;
    if (froxel_coord.x >= pc.froxel_dimensions.x || froxel_coord.y >= pc.froxel_dimensions.y)
        return;

    SamplerState samp = get_sampler<SamplerState>(LINEAR_CLAMP_SAMPLER);
    Texture3D light_scattering_tex = get_sampled_image<Texture3D>(pc.light_scattering_tex);
    RWTexture3D integrated_light_scattering_tex = get_storage_image<RWTexture3D>(pc.integrated_light_scattering_tex);

    float3 integrated_scattering = float3(0.0);
    float integrated_transmission = 1.0;

    float current_z = 0.0;

    float3 rcp_froxel_dimensions = 1.0 / float3(pc.froxel_dimensions);

    uint froxel_z = pc.froxel_dimensions.z;
    for (uint z = 0; z < froxel_z; z++)
    {
        froxel_coord.z = z;

        float next_z = pc.near * pow(pc.far / pc.near, (z + 1.0 + 0.5) / float(froxel_z)); // no negation as we want +ve linear depth
        float z_step = next_z - current_z; // removed abs
        current_z = next_z;

        // from Physically Based Sky, Atmosphere and Cloud Rendering by Hillaire
        float4 scattering_extinction = light_scattering_tex.SampleLevel(samp, (float3(froxel_coord) + float3(0.5)) / float3(pc.froxel_dimensions), 0.0);
        float3 scattering = scattering_extinction.rgb;
        float extinction = scattering_extinction.a;
        float clamped_extinction = max(extinction, 0.00001);

        float transmittance = exp(-extinction * z_step); // beer's law
        scattering = (scattering - (scattering * transmittance)) / clamped_extinction;

        integrated_scattering += scattering * integrated_transmission;
        integrated_transmission *= transmittance;

        integrated_light_scattering_tex.Store(uint3(froxel_coord), float4(integrated_scattering, integrated_transmission));
    }
}
