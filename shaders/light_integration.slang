import bindings;
import volumetrics;

struct PushConstants
{
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

    Texture3D light_scattering_tex = get_sampled_image<Texture3D>(pc.light_scattering_tex);
    RWTexture3D integrated_light_scattering_tex = get_storage_image<RWTexture3D>(pc.integrated_light_scattering_tex);

    float3 integrated_scattering = float3(0.0);
    float integrated_transmission = 1.0;

    uint num_slices = pc.froxel_dimensions.z;
    for (uint slice = 0; slice < num_slices; slice++)
    {
        froxel_coord.z = slice;

        // TODO: this can be optimized so we don't recalculate the nearer froxel
        float view_z0 = view_z_from_slice(slice, num_slices, pc.near, pc.far);
        float view_z1 = view_z_from_slice(slice + 1, num_slices, pc.near, pc.far);
        float d = view_z1 - view_z0;

        // from Physically Based Sky, Atmosphere and Cloud Rendering by Hillaire
        float4 scattering_extinction = light_scattering_tex.Load(int4(froxel_coord, 0));
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
