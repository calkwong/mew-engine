import bindings;
import types;
import util;
import prefix_sum;

struct DrawCommands
{
	uint opaque_count;
	uint alpha_clip_count;

	DrawCommand commands[1000000];
};

struct PushConstants
{
	float4x4 view;
	float4 frustum_planes;

	ObjectData* object_buffer;
	MeshData* mesh_buffer;
	uint* mesh_indices_buffer;
	DrawCommands* draw_buffer;
	Dispatch* dispatch_buffer;
	uint* mesh_visibility_buffer;
	PrefixSum* prefix_sum_buffer;

	uint count;
	uint late;
	uint texture_id;
	uint occlusion_enabled;

	float p00;
	float p11;
	float near;
	float far; // this is draw distance

	float2 resolution;
	float texture_lod;
	float lod_distance_factor;
	uint lod_enabled;
	uint task_submit;
	uint post_pass;
};

[[vk::push_constant]] PushConstants pc;

// 2D Polyhedral Bounds of a Clipped, Perspective-Projected 3D Sphere. Michael Mara, Morgan McGuire. 2013
// https://zeux.io/2023/01/12/approximate-projected-bounds/
bool project_sphere(float3 c, float r, float near, float p00, float p11, inout float4 aabb)
{
	if (c.z < r + near)
		return false;

    float3 cr = c * r;
    float czr2 = c.z * c.z - r * r;

    float vx = sqrt(c.x * c.x + czr2);
    float minx = (vx * c.x - cr.z) / (vx * c.z + cr.x);
    float maxx = (vx * c.x + cr.z) / (vx * c.z - cr.x);

    float vy = sqrt(c.y * c.y + czr2);
    float miny = (vy * c.y - cr.z) / (vy * c.z + cr.y);
    float maxy = (vy * c.y + cr.z) / (vy * c.z - cr.y);

    aabb = float4(minx * p00, miny * p11, maxx * p00, maxy * p11);
    // clip space -> uv space
    aabb = aabb.xwzy * float4(0.5, -0.5, 0.5, -0.5) + float4(0.5);

    return true;
}

bool frustum_culled(float3 center, float radius)
{
	bool visible = true;

	// symmetrical sphere-plane distance test
	visible = visible && (center.z * pc.frustum_planes.y - abs(center.x) * pc.frustum_planes.x) >= -radius;
	visible = visible && (center.z * pc.frustum_planes.w - abs(center.y) * pc.frustum_planes.z) >= -radius;
	visible = visible && (center.z - radius <= -pc.near) && (center.z + radius >= -pc.far);

	return !visible;
}

bool occlusion_culled(float3 center, float radius)
{
	float4 aabb;

	// TODO: do we need this conditional check?
	if (project_sphere(float3(center.x, center.y, -center.z), radius, pc.near, pc.p00, pc.p11, aabb))
	{
		float2 pixel_size = (aabb.zw - aabb.xy) * pc.resolution;

		float max_dim = max(pixel_size.x, pixel_size.y);

		float mip = ceil(log2(max_dim));
		mip = min(mip, pc.texture_lod);
		SamplerState samp = get_sampler<SamplerState>(DEPTH_REDUCTION_SAMPLER);
		float depth = get_sampled_image<Texture2D>(pc.texture_id).SampleLevel(samp, (aabb.zw + aabb.xy) * 0.5, mip).r;

		float depth_sphere = pc.near / (-center.z - radius);

		return depth_sphere < depth; // reverse depth
	}

	return false;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    uint3 global_id : SV_DispatchThreadID
)
{

	if (global_id.x >= pc.count)
		return;

	uint id = pc.mesh_indices_buffer[global_id.x];

	bool late = pc.late == 1;

	// TODO: mesh_visibility_buffer using 4 bytes instead of 1 bit for visibility
	// TODO: can potentially skip this for post_pass != 0 (masked, transparent)?
	bool visible_last_frame = pc.mesh_visibility_buffer[id] == 1;

	if (!late && !visible_last_frame)
		return;

	ObjectData obj = pc.object_buffer[id];

	uint mesh_id = obj.mesh_id;
	MeshData mesh = pc.mesh_buffer[mesh_id];

	float3 mpos = mesh.center;
	float3 center = rotate_quat(mpos, obj.orientation) * obj.scale + obj.translation; // world space
	center = mul(pc.view, float4(center, 1.0)).xyz; // view space

	float radius = mesh.radius;

    // non uniform scaling not support
	float max_scale = obj.scale;
	radius = radius * max_scale;

	bool visible = !frustum_culled(center, radius);
	bool should_draw = visible;

	if (late && visible)
	{
		if (pc.occlusion_enabled == 1)
			visible = visible && !occlusion_culled(center, radius);
		// TODO: verify we can remove postpass check here as we dispatch by geometry type?
		should_draw = (pc.task_submit == 0) ? visible && (!visible_last_frame || pc.post_pass != 0) : visible;
	}

	if (should_draw)
	{

		uint index = 0;
		if (pc.lod_enabled == 1) // TODO: move above into this block?
		{
			// automatic LOD selection - at distance d, how many world space units does each pixel cover?
			// is the absolute error computed (simplifyscale * result_error) less visible than error above? if so, move onto higher LOD
			float d = max(0.0, length(center) - radius);
			float e = d * pc.lod_distance_factor / max_scale;

			for (uint i = 0; i < mesh.lod_count; i++)
			{
				index = e >= mesh.lods[i].error ? i : index;
			}
		}

		if (pc.task_submit == 1)
		{
			uint meshlet_count = mesh.lods[index].meshlet_count;
			uint meshlet_offset = mesh.lods[index].meshlet_offset;

			if (late)
			{
    			uint mesh_visible_last_frame = uint(visible_last_frame) << 31;
    			meshlet_offset = meshlet_offset | mesh_visible_last_frame;
			}

			// uint count = atomicAdd(dispatch_buffer.workgroup_x, meshlet_count);
			uint count;
			InterlockedAdd(pc.dispatch_buffer.workgroup_x, meshlet_count, count);

			prefix_sum_inclusive_append(pc.prefix_sum_buffer, id, meshlet_count, meshlet_offset);

		}
		else
		{
			// TODO: currently, third pass for masked geometry is only culling for late==1. it is incremented in opaqueCount. make a proper distinction when we cull masked geometry in early and late?
			uint count;
			InterlockedAdd(pc.draw_buffer.opaque_count, 1, count);

			DrawCommand draw;
			draw.index_count = mesh.lods[index].count;
			draw.instance_count = 1;
			draw.first_index = mesh.lods[index].first_index;
			draw.vertex_offset = mesh.vertex_offset;
			draw.first_instance = id;

			pc.draw_buffer.commands[count] = draw;
		}
	}

	if (late)
		pc.mesh_visibility_buffer[id] = uint(visible);
}
