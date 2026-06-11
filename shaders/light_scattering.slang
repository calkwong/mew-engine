import bindings;
import util;
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
};

[[vk::push_constant]]
PushConstants pc;

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

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    uint3 froxel_coord = global_id;
    if (froxel_coord.x >= pc.froxel_dimensions.x || froxel_coord.y >= pc.froxel_dimensions.y || froxel_coord.z >= pc.froxel_dimensions.z)
        return;

    SamplerState samp = get_sampler<SamplerState>(NEAREST_SAMPLER);
    Texture3D scattering_extinction_tex = get_sampled_image<Texture3D>(pc.scattering_extinction_tex);
    RWTexture3D light_scattering_tex = get_storage_image<RWTexture3D>(pc.light_scattering_tex);

    float3 world_pos = world_from_froxel(froxel_coord);
    float3 rcp_froxel_dimensions = 1.0 / float3(pc.froxel_dimensions);
    float3 fog_data_uvw = float3(froxel_coord) * rcp_froxel_dimensions;

    float4 scattering_extinction = scattering_extinction_tex.SampleLevel(samp, fog_data_uvw, 0.0);
    float extinction = scattering_extinction.a;

    float3 lighting = float3(0.0);

    // TODO: apply phase function, ambient, figure out if we apply beer's law here
    if (extinction > 0.01)
    {
        uint cascade_index = 0;
        float shadow = calculate_shadow(world_pos, cascade_index, pc.shadowmap_id);
        // float shadow = 1.0;
        float3 sunlight = shadow * uniforms.sunlight_color.xyz;
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

            point_lights += light_color * attenuation;
        }
#endif

        lighting += (sunlight + point_lights);
    }

    float3 scattering = scattering_extinction.rgb * lighting;
    light_scattering_tex.Store(froxel_coord, float4(scattering, extinction));
}
