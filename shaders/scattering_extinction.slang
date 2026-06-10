import bindings;
import util;

struct PushConstants
{
    float4x4 inverse_view_proj;
    uint3 froxel_dimensions;
    float near;
    float far;
    uint noise_tex;
    uint scattering_extinction_tex;
    uint volumetric_noise_pos_mult; // [0, 1]
    uint volumetric_noise_speed_mult; // [0, 1]
    uint current_frame;
    float fog_density_modifier; // [0, 1]
    float height_fog_density_modifier; // [0, 10]
    float scattering_factor; // [0, 1]
    float height_fog_falloff; // [0, 10]
};

[[vk::push_constant]]
PushConstants pc;

// TODO: going with 160x90x128 FROXEL dimensions for now, experiment with different groupsizes
#define FROXEL_X 32
#define FROXEL_Y 32
#define FROXEL_Z 1

float linear_depth_to_raw_depth(float linear_depth, float near, float far)
{
    return (near * far) / (linear_depth * (near - far)) - far / (near - far);
}

// infinite far plane + reverse-z
float linear_depth_to_raw_depth(float linear_depth, float near)
{
    return near / linear_depth;
}

float3 world_from_froxel(uint3 coord)
{
    float2 uv = (float2(coord.xy) + float2(0.5)) / float2(pc.froxel_dimensions.xy);
    float froxel_z = float(pc.froxel_dimensions.z);
    float slice = float(coord.z);

    // from ID tech
    float linear_depth = pc.near * pow(pc.far / pc.near, (slice + 0.5) / froxel_z); // no negation as we want +ve linear depth
    // float depth = linear_depth_to_raw_depth(linear_depth, pc.far, pc.near); // reverse-z
    float depth = linear_depth_to_raw_depth(linear_depth, pc.near); // infinite far plane + reverse z

    return reconstruct_world_pos(depth, pc.inverse_view_proj, uv);
}

float4 scattering_extinction_from_color_density(float3 color, float density)
{
    float extinction = density * pc.scattering_factor;
    return float4(color * extinction, extinction);
}

[shader("compute")]
[numthreads(FROXEL_X, FROXEL_Y, FROXEL_Z)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    // TODO: use NEAREST_REPEAT_SAMPLER
    SamplerState samp = get_sampler<SamplerState>(NEAREST_SAMPLER);
    Texture3D<float> noise_tex = get_sampled_image<Texture3D<float>>(pc.noise_tex);
    RWTexture3D scattering_extinction_tex = get_storage_image<RWTexture3D>(pc.scattering_extinction_tex);

    float4 scattering_extinction = float4(0.0);

    uint3 froxel_coord = global_id;
    float3 world_pos = world_from_froxel(froxel_coord);

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

    scattering_extinction_tex.Store(global_id, scattering_extinction);
}
