import bindings;
import types;
import util;
import prefix_sum;
import culling;

struct PushConstants
{
    float4x4 view;
    float4 frustum_planes;

    ObjectData* object_buffer;
    Meshlet* meshlet_buffer;
    uint* cluster_indices_buffer;
    Dispatch* meshlet_dispatch_buffer;
    uint* meshlet_visibility_buffer;
    PrefixSum* prefix_sum_buffer;

    uint count; // unused?
    uint is_late;
    uint texture_id;
    uint occlusion_enabled;

    float p00;
    float p11;
    float near;
    float far; // this is draw distance

    float2 depth_pyramid_extent;
    float2 depth_buffer_extent;
    float texture_lod;
    float lod_distance_factor;
    uint lod_enabled;
    uint task_submit;
    uint post_pass;
};

[[vk::push_constant]]
PushConstants pc;

bool occlusion_culled(float3 center, float radius)
{
	float4 aabb;

	// TODO: do we need this conditional check?
	if (project_sphere(float3(center.x, center.y, -center.z), radius, pc.near, pc.p00, pc.p11, aabb))
    {
        // Assumption here is that depth buffer is non PoT, and mip 0 of HZB is nearest rounded down PoT
        float2 valid_extent = pc.depth_buffer_extent * 0.5;
        float2 hzb_uv_scale = valid_extent / pc.depth_pyramid_extent;

        float mip = get_occlusion_mip(aabb, valid_extent, pc.texture_lod);
        SamplerState samp = get_sampler<SamplerState>(DEPTH_REDUCTION_SAMPLER);

        float2 adjusted_uv = (aabb.zw + aabb.xy) * 0.5 * hzb_uv_scale;
        float depth = get_sampled_image<Texture2D>(pc.texture_id).SampleLevel(samp, adjusted_uv, mip).r;

        float depth_sphere = pc.near / (-center.z - radius);

        return depth_sphere < depth; // reverse depth
    }

	return false;
}

#define WG_SIZE_X 256

[shader("compute")]
[numthreads(WG_SIZE_X, 1, 1)]
void main(
    uint t_index: SV_GroupIndex,
    uint3 wg_id: SV_GroupID
)
{
    uint ti = wg_id.x * WG_SIZE_X + t_index;

    uint64_t counter = get_prefix_sum_counter(pc.prefix_sum_buffer);
    uint total_meshlet_count = uint(counter);

    uint prefix_sum_idx = prefix_sum_binary_search(pc.prefix_sum_buffer, ti);
    PrefixSumData prefix_sum_data = pc.prefix_sum_buffer.data[prefix_sum_idx];
    uint meshlet_lod_offset = prefix_sum_data.lod_offset; // NOT final meshlet id

    uint mesh_visible_last_frame;
    bool late = pc.is_late == 1;
    if (late)
    {
        mesh_visible_last_frame = bitfieldExtract(meshlet_lod_offset, 31u, 1u); // careful order matters!
        meshlet_lod_offset = bitfieldExtract(meshlet_lod_offset, 0u, 31u);
    }

    uint object_id = prefix_sum_data.instance_id;

    bool overflow = prefix_sum_idx == total_meshlet_count;

    uint inclusive_sum = overflow ? total_meshlet_count : pc.prefix_sum_buffer.data[prefix_sum_idx + 1].sum;
    uint exclusive_sum = prefix_sum_data.sum;
    uint local_meshlet_count = inclusive_sum - exclusive_sum;

    // get meshlet index
    uint local_meshlet_index = ti - (inclusive_sum - local_meshlet_count);

    if (ti < total_meshlet_count)
    {
        ObjectData obj = pc.object_buffer[object_id];
        uint meshlet_visibility_offset = obj.meshlet_bit_offset + local_meshlet_index;

        // bit shift in place of floor division, then bit& to extract the relevant bit
        uint meshlet_visibility_bit = pc.meshlet_visibility_buffer[meshlet_visibility_offset >> 5] & (1u << (meshlet_visibility_offset & 31));

        bool meshlet_visible_last_frame = meshlet_visibility_bit != 0;

        if (!late && !meshlet_visible_last_frame)
            return;

        bool skip = false; // always false for early pass
        if (late && meshlet_visible_last_frame && mesh_visible_last_frame == 1)
            skip = true;
        // TODO: update this when we start culling postpass > 0 for late==0
        // take into consideration both meshlet and mesh_visibility_last_frame when updating this in the future
        // we never skip mask and transparent (postpass != 0) regardless of visibility last frame as they only run for late==1
        if (pc.post_pass != 0)
            skip = false;

        // reprocess all meshlets of visible meshes
        Meshlet meshlet = pc.meshlet_buffer[local_meshlet_index + meshlet_lod_offset];

        float3 mpos = float3(meshlet.cx, meshlet.cy, meshlet.cz);
        float3 center = rotate_quat(mpos, obj.orientation) * obj.scale + obj.translation; // world space
        center = mul(pc.view, float4(center, 1.0)).xyz; // view space

        float radius = meshlet.radius;

        // non uniform scaling not supported
        float max_scale = obj.scale;
        radius = radius * max_scale;

        // meshlet cone culling
        // float3 cone_axis = float3(int(meshlet.cone_axis[0]), int(meshlet.cone_axis[1]), int(meshlet.cone_axis[2]));
        // cone_axis /= 127.0;
        // cone_axis = rotate_quat(cone_axis, obj.orientation);
        // cone_axis = mul(float3x3(view), cone_axis);

        // float3 bound_sphere_dir = center - camera_pos;
        // float bound_sphere_distance = length(bound_sphere_dir);
        // float cone_cutoff = int(meshlet.cone_cutoff) / 127.0;
        // cone_cutoff = cone_cutoff * bound_sphere_distance + radius;

        // bool visible = dot(bound_sphere_dir, cone_axis) < cone_cutoff;

        // frustum culling
        bool visible = !frustum_culled(center, radius, pc.frustum_planes, pc.near, pc.far);

        if (late && visible)
        {
            if (pc.occlusion_enabled == 1)
            {
                visible = visible && !occlusion_culled(center, radius);
            }
        }

        // render or not - both pass 1 & 2
        if (visible && !skip)
        {
            // similar performance with subgroupops, maybe compiler already optimized this?
            uint cluster_index;
            InterlockedAdd(pc.meshlet_dispatch_buffer.workgroup_x, 1, cluster_index);
            pc.meshlet_dispatch_buffer.workgroup_y = 1;
            pc.meshlet_dispatch_buffer.workgroup_z = 1;

            pc.cluster_indices_buffer[cluster_index] = ti;
        }

        if (late)
        {
            // TODO: potentially slow atomic
            // divide offset by 32, producing index in the buffer that holds the bit we want
            // extract the bit via (offset & 31); equivalent to modulo 32
            if (visible)
                InterlockedOr(pc.meshlet_visibility_buffer[meshlet_visibility_offset >> 5], 1u << (meshlet_visibility_offset & 31));
            else
                InterlockedAnd(pc.meshlet_visibility_buffer[meshlet_visibility_offset >> 5], ~(1u << (meshlet_visibility_offset & 31)));
        }
    }
}
