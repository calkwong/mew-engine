import bindings;
import util;
import volumetrics;
import shadows;
import types;
import pbr;

struct PushConstants
{
    float4x4 inverse_view_proj;
    PointLight* light_buffer;
    uint* light_index_buffer;
    LightGrid* light_grid_buffer;
    float near;
    float far;

    uint3 froxel_dimensions;
    uint scattering_extinction_tex;

    uint light_scattering_tex;
    uint shadowmap_id;
    float light_cluster_scale;
    float light_cluster_bias;

    float2 pixels_per_cluster;
    float2 halton;

    float phase_anisotropy;
    uint blue_noise_tex;
    uint current_frame;
    uint point_lights;

    uint previous_light_scattering_tex;
    float volumetrics_scale;
    float volumetrics_bias;
    uint first_frame;
    uint temporal_filter;
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

    SamplerState samp = get_sampler<SamplerState>(LINEAR_CLAMP_SAMPLER);
    Texture3D scattering_extinction_tex = get_sampled_image<Texture3D>(pc.scattering_extinction_tex);
    Texture3D previous_light_scattering_tex = get_sampled_image<Texture3D>(pc.previous_light_scattering_tex);
    RWTexture3D light_scattering_tex = get_storage_image<RWTexture3D>(pc.light_scattering_tex);

    float temporal_reprojection_jitter_scale = 0.5; // TODO: make this a cvar
    float3 world_pos = world_from_froxel(froxel_coord, pc.froxel_dimensions, pc.inverse_view_proj, pc.near, pc.far, pc.halton, temporal_reprojection_jitter_scale, pc.blue_noise_tex, pc.current_frame);
    float3 rcp_froxel_dimensions = 1.0 / float3(pc.froxel_dimensions);
    float3 fog_data_uvw = float3(froxel_coord) * rcp_froxel_dimensions;

    float4 scattering_extinction = scattering_extinction_tex.SampleLevel(samp, fog_data_uvw, 0.0);
    float extinction = scattering_extinction.a;

    float3 lighting = float3(0.0);

    // TODO: ambient
    {
        float3 V = normalize(uniforms.camera_pos.xyz - world_pos);
        float3 L = normalize(uniforms.sunlight_dir.xyz);
        float VdotL = dot(V, L);
        float phase = phase_shlick(pc.phase_anisotropy, VdotL);

        uint cascade_index = 0;
        float shadow = calculate_shadow(world_pos, cascade_index, pc.shadowmap_id);
        float3 sunlight = shadow * uniforms.sunlight_color.xyz * phase;
        float sunlight_intensity = 10.0; // for debugging
        sunlight *= sunlight_intensity;

        // this is a hack - to properly implement we need to implement density here rather than in an earlier pass and
        // looking up a texture3D, which is way too expensive as we need to convert from world pos to froxel uv for each step
        float accum_density = 0.0;
        float step_size = 5.0;
        for (int i = 0; i < 5; i++)
        {
            accum_density += extinction * step_size;
        }
        sunlight *= exp(-accum_density);

        float3 point_lights = float3(0.0);

        // interate over local lights in light cluster
        if (pc.point_lights == 1)
        {
            float view_z = mul(uniforms.view, float4(world_pos, 1.0)).z;
            uint light_slice = get_cluster_z_slice(-view_z, pc.light_cluster_scale, pc.light_cluster_bias);
            uint cluster_index = get_cluster_index(float2(froxel_coord.x, pc.froxel_dimensions.y - froxel_coord.y), pc.pixels_per_cluster.xy, light_slice);

            LightGrid grid = pc.light_grid_buffer[cluster_index];
            uint offset = grid.offset;
            uint local_light_count = grid.count;

            for (int i = 0; i < local_light_count; i++)
            {
                uint index = pc.light_index_buffer[offset + i];

                PointLight point_light = pc.light_buffer[index];
                float3 light_pos = point_light.pos.xyz;
                float light_radius = point_light.pos.w; // unused, taking into account falloff attenuation?
                float3 light_color = point_light.color.xyz;
                // float intensity = 1.0; // for debugging
                // light_color *= intensity;

                light_pos = mul(uniforms.light_rot, float4(light_pos, 1.0)).xyz;
                L = light_pos - world_pos;

                float attenuation = get_square_falloff_attenuation(L, light_radius);

                light_color *= exp(-extinction * length(L));

                L = normalize(L);
                VdotL = dot(L, V);
                float phase = phase_shlick(pc.phase_anisotropy, VdotL);
                point_lights += light_color * attenuation * phase;
            }
        }

        lighting += sunlight + point_lights;
    }

    float3 scattering = scattering_extinction.rgb * lighting;

    if (pc.first_frame == 1)
    {
        light_scattering_tex.Store(froxel_coord, float4(scattering, extinction));
        return;
    }

    if (pc.temporal_filter == 1)
    {
        float4 history_clip = mul(uniforms.prev_view_proj, float4(world_pos, 1.0));
        float3 history_ndc = history_clip.xyz / history_clip.w;
        float history_linear_depth = pc.near / history_ndc.z;
        uint history_slice = uint(floor(log(history_linear_depth) * pc.volumetrics_scale - pc.volumetrics_bias));
        // uint max_slice = pc.froxel_dimensions.z - 1;
        uint max_slice = pc.froxel_dimensions.z;
        history_slice = clamp(history_slice, 0, max_slice);
        float w = float(history_slice) / float(max_slice);

        float3 history_uvw = float3(history_ndc.xy * float2(0.5, -0.5) + float2(0.5), w);

        if (all(history_uvw >= 0.0) && all(history_uvw <= 1.0))
        {
            float4 history = previous_light_scattering_tex.SampleLevel(samp, history_uvw, 0.0);

            // history = max(history, float4(scattering, extinction));
            const float reprojection_factor = 0.05;
            scattering = lerp(history.rgb, scattering, reprojection_factor);
            extinction = lerp(history.a, extinction, reprojection_factor);
        }
    }

    light_scattering_tex.Store(froxel_coord, float4(scattering, extinction));
}
