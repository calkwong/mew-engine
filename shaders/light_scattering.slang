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

    float2 cluster_dim;
    float phase_anisotropy;
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
    if (froxel_coord.x >= pc.froxel_dimensions.x || froxel_coord.y >= pc.froxel_dimensions.y || froxel_coord.z >= pc.froxel_dimensions.z)
        return;

    SamplerState samp = get_sampler<SamplerState>(LINEAR_CLAMP_SAMPLER);
    Texture3D scattering_extinction_tex = get_sampled_image<Texture3D>(pc.scattering_extinction_tex);
    RWTexture3D light_scattering_tex = get_storage_image<RWTexture3D>(pc.light_scattering_tex);

    float3 world_pos = world_from_froxel(froxel_coord, pc.froxel_dimensions, pc.inverse_view_proj, pc.near, pc.far);
    float3 rcp_froxel_dimensions = 1.0 / float3(pc.froxel_dimensions);
    float3 fog_data_uvw = float3(froxel_coord) * rcp_froxel_dimensions;

    float4 scattering_extinction = scattering_extinction_tex.SampleLevel(samp, fog_data_uvw, 0.0);
    float extinction = scattering_extinction.a;

    float3 lighting = float3(0.0);

    // TODO: ambient, figure out if we apply beer's law here
    // if (extinction > 0.01)
    {
        float3 V = normalize(uniforms.camera_pos.xyz - world_pos);
        float phase = phase_shlick(pc.phase_anisotropy, dot(normalize(uniforms.sunlight_dir.xyz), V));

        uint cascade_index = 0;
        float shadow = calculate_shadow(world_pos, cascade_index, pc.shadowmap_id);
        float3 sunlight = shadow * float3(10.0) * phase;
        // float3 sunlight = shadow * uniforms.sunlight_color.xyz * phase;
        // sunlight *= exp(-extinction); // this is a simplification, we should march along sun direction

        // interate over local lights in light cluster
        float view_z = mul(uniforms.view, float4(world_pos, 1.0)).z;
        uint light_slice = get_cluster_z_slice(view_z, pc.light_cluster_scale, pc.light_cluster_bias);
        uint cluster_index = get_cluster_index(float2(froxel_coord.x, pc.froxel_dimensions.y - froxel_coord.y), pc.cluster_dim.xy, light_slice);

        LightGrid grid = pc.light_grid_buffer[cluster_index];
        uint offset = grid.offset;
        uint local_light_count = grid.count;
        float3 point_lights = float3(0.0);

#if 1
        for (int i = 0; i < local_light_count; i++)
        {
            uint index = pc.light_index_buffer[offset + i];

            PointLight point_light = pc.light_buffer[index];
            float3 light_pos = point_light.pos.xyz;
            float light_radius = point_light.pos.w; // unused, taking into account falloff attenuation?
            float3 light_color = point_light.color.xyz;

            light_pos = mul(uniforms.light_rot, float4(light_pos, 1.0)).xyz;
            float3 distance = light_pos - world_pos;

            float attenuation = get_square_falloff_attenuation(distance, light_radius);

            // light_color *= exp(-extinction * length(distance));

            float phase = phase_shlick(pc.phase_anisotropy, dot(normalize(distance), V));
            point_lights += light_color * attenuation * phase;
        }
#endif

        lighting += sunlight + point_lights;
    }

    float3 scattering = scattering_extinction.rgb * lighting;
    light_scattering_tex.Store(froxel_coord, float4(scattering, extinction));
}
