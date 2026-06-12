import bindings;
import volumetrics;

// this shader could be combined with light_scattering

struct PushConstants
{
    float4x4 inverse_view_proj;
    uint3 froxel_dimensions;
    float near;
    float far;
    uint noise_tex;
    uint scattering_extinction_tex;
    float volumetric_noise_pos_mult; // [0, 1]
    float volumetric_noise_speed_mult; // [0, 1]
    uint current_frame;
    float fog_density_modifier; // [0, 1]
    float height_fog_density_modifier; // [0, 10]
    float scattering_factor; // [0, 1]
    float height_fog_falloff; // [0, 10]
};

[[vk::push_constant]]
PushConstants pc;

float4 scattering_extinction_from_color_density(float3 color, float density)
{
    float extinction = density * pc.scattering_factor;
    return float4(color * extinction, extinction);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    uint3 froxel_coord = global_id;
    if (froxel_coord.x >= pc.froxel_dimensions.x || froxel_coord.y >= pc.froxel_dimensions.y || froxel_coord.z >= pc.froxel_dimensions.z)
        return;

    SamplerState samp = get_sampler<SamplerState>(NEAREST_REPEAT_SAMPLER);
    Texture3D<float> noise_tex = get_sampled_image<Texture3D<float>>(pc.noise_tex);
    RWTexture3D scattering_extinction_tex = get_storage_image<RWTexture3D>(pc.scattering_extinction_tex);

    float4 scattering_extinction = float4(0.0);

    float3 world_pos = world_from_froxel(froxel_coord, pc.froxel_dimensions, pc.inverse_view_proj, pc.near, pc.far);

    // large (small) pos_mult -> smaller (larger), more detailed (smooth) fog?
    // float3(1.0, 0.1, 2.0) behaves like wind dir
    // note: w points into screen and is negative
    float3 sample_uvw = world_pos * pc.volumetric_noise_pos_mult + float3(1.0, 0.1, 2.0) * pc.current_frame * pc.volumetric_noise_speed_mult;
    float fog_noise = noise_tex.SampleLevel(samp, sample_uvw, 0.0).r;
    fog_noise = saturate(fog_noise * fog_noise); // why?

    // add constant fog
    float fog_density = pc.fog_density_modifier * fog_noise;
    scattering_extinction += scattering_extinction_from_color_density(float3(0.5), fog_density);

    // add height fog: fog decreases exponentially with elevation
    float height_fog = pc.height_fog_density_modifier * exp(-pc.height_fog_falloff * max(world_pos.y, 0.0)) * fog_noise;
    scattering_extinction += scattering_extinction_from_color_density(float3(0.5), height_fog);

    scattering_extinction_tex.Store(froxel_coord, scattering_extinction);
}
