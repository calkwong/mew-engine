import bindings;
import volumetrics;

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
    if (any(froxel_coord >= pc.froxel_dimensions))
        return;

    SamplerState linear_clamp_samp = get_sampler<SamplerState>(LINEAR_CLAMP_SAMPLER);
    Texture3D light_scattering_tex = get_sampled_image<Texture3D>(pc.light_scattering_tex);
    RWTexture3D integrated_light_scattering_tex = get_storage_image<RWTexture3D>(pc.integrated_light_scattering_tex);

    float3 integrated_scattering = float3(0.0);
    float integrated_transmission = 1.0;

    float3 prev_pos = uniforms.camera_pos.xyz;
    uint froxel_z = pc.froxel_dimensions.z;
    for (uint z = 0; z < froxel_z; z++)
    {
        froxel_coord.z = z;

        float3 curr_pos = world_from_froxel_no_jitter(froxel_coord, pc.froxel_dimensions, pc.inverse_view_proj, pc.near, pc.far);
        float d = distance(curr_pos, prev_pos);
        prev_pos = curr_pos;

        // from Physically Based Sky, Atmosphere and Cloud Rendering by Hillaire
        float4 scattering_extinction = light_scattering_tex.SampleLevel(linear_clamp_samp, (float3(froxel_coord) + float3(0.5)) / float3(pc.froxel_dimensions), 0.0);
        float3 scattering = scattering_extinction.rgb;
        float extinction = scattering_extinction.a;
        float clamped_extinction = max(extinction, 0.00001);

        float transmittance = exp(-extinction * d);
        scattering = (scattering - (scattering * transmittance)) / clamped_extinction;
        integrated_scattering += scattering * integrated_transmission;
        integrated_transmission *= transmittance;

        integrated_light_scattering_tex.Store(froxel_coord, float4(integrated_scattering, integrated_transmission));
    }
}
