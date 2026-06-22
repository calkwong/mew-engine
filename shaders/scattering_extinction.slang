import bindings;
import volumetrics;

struct PushConstants
{
    float4x4 inverse_view_proj;
    uint3 froxel_dimensions;
    uint current_frame;
    float near;
    float far;
    uint perlin_noise_tex;
    uint scattering_extinction_tex;
    float volumetric_noise_pos_mult;
    float volumetric_noise_speed_mult;
    float fog_density_modifier;
    float height_fog_density_modifier;
    float scattering_factor;
    float height_fog_falloff;
    uint frame_index;
};

[[vk::push_constant]]
PushConstants pc;

float4 scattering_extinction_from_color_density(float3 color, float density)
{
    float extinction = density * pc.scattering_factor;
    return float4(color * extinction, extinction);
}

float sd_box(float3 p, float3 b)
{
    float3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

[shader("compute")]
[numthreads(8, 8, 8)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    uint3 froxel_coord = global_id;
    if (any(froxel_coord >= pc.froxel_dimensions))
        return;

    // SamplerState samp = get_sampler<SamplerState>(LINEAR_SAMPLER);
    // Texture3D<float> noise_tex = get_sampled_image<Texture3D<float>>(pc.perlin_noise_tex);
    RWTexture3D scattering_extinction_tex = get_storage_image<RWTexture3D>(pc.scattering_extinction_tex);

    float4 scattering_extinction = float4(0.0);

    float3 world_pos = jittered_world_from_froxel(froxel_coord, pc.froxel_dimensions, pc.inverse_view_proj, pc.near, pc.far, pc.current_frame);

    // large (small) pos_mult -> smaller (larger), more detailed (smooth) fog?
    // float3(1.0, 0.1, 2.0) behaves like wind dir
    // note: w points into screen and is negative
    // note: this eq seems slightly broken for some combinations
    // float3 sample_uvw = world_pos * pc.volumetric_noise_pos_mult + float3(1.0, 0.1, 2.0) * pc.current_frame * pc.volumetric_noise_speed_mult;
    // float fog_noise = noise_tex.SampleLevel(samp, sample_uvw, 0.0).r;
    // fog_noise = saturate(fog_noise * fog_noise); // why?

    // add depth fog
    float dist_meters = length(world_pos - uniforms[pc.frame_index].camera_pos.xyz) * 0.01;
    float depth_density = 1.0 - exp(-dist_meters * pc.fog_density_modifier);
    scattering_extinction += scattering_extinction_from_color_density(float3(0.5), depth_density);

    // add height fog
    const float base_height_start = 0.0;
    float height_delta_meters = max((world_pos.y - base_height_start) * 0.01, 0.0);
    float height_density = exp(-height_delta_meters * pc.height_fog_density_modifier * pc.height_fog_falloff);
    height_density = height_density != 1.0 ? height_density : 0.0;
    if (height_density != 1.0)
        scattering_extinction += scattering_extinction_from_color_density(float3(1.0), height_density);

    float cube_fog = 1.0 - smoothstep(0.0, 0.25, sd_box(float3(world_pos - float3(3.0, 1.0, 1.75)), float3(0.75)));
    if (cube_fog != 0.0)
    {
        // TODO: volume scattering factor?
        scattering_extinction.rgb += float3(1., 0., 0.) * cube_fog; // tinted red
        scattering_extinction.a += cube_fog;
    }

    // TODO: weighted phaseG and emission?

    scattering_extinction_tex.Store(froxel_coord, scattering_extinction);
}
