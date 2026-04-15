import bindings;
import types;
import util;

struct ShadowDrawCommands
{
    DrawCommands draws[4];
};

struct PushConstants
{
	ObjectData* object_buffer;
	MeshData* mesh_buffer;
	uint* mesh_indices_buffer;
	ShadowDrawCommands* draw_buffer;
	uint count;
	uint lod_enabled;
};

[[vk::push_constant]] PushConstants pc;

bool partially_inside_cascade(float3 center_world, float radius, uint cascade_index)
{
	float3 center_view = mul(uniforms.shadow_views[cascade_index], float4(center_world, 1.0)).xyz; // light space
	float cascade_radius = uniforms.shadow_widths[cascade_index];

	bool visible = true;

	visible = visible && ( (abs(center_view.x) - radius) <= cascade_radius );
    visible = visible && ( (abs(center_view.y) - radius) <= cascade_radius );
	visible = visible && (center_view.z + radius >= (-2.0 * cascade_radius)); // no near plane check as we have depthclamp==true

	return visible;
}

// idea from: https://github.com/expenses/lighthugger/blob/9d6c46b3e339e9673b69b7d9933d6a439051232c/src/shaders/cull_instances.comp
// TODO: is this even worth it? unsure if implementation is robust, requires more testing. disabled for now.
bool completely_inside_cascade(float3 center_world, float radius, uint cascade_index)
{
    float3 center_view = mul(uniforms.shadow_views[cascade_index], float4(center_world, 1.0)).xyz;
    float cascade_radius = uniforms.shadow_widths[cascade_index];

	bool visible = true;

	visible = visible && ( (abs(center_view.x) + radius) <= cascade_radius );
    visible = visible && ( (abs(center_view.y) + radius) <= cascade_radius );
    visible = visible && (center_view.z - radius >= (-2.0 * cascade_radius)); // no near plane check as we have depthclamp==true

    if (visible)
    {
        bool inside = center_view.z + radius <= (-2.0 * cascade_radius); // tailored to depthclamp==true
        return inside;
    }

    return visible;
}

[shader("compute")]
[numthreads(256, 4, 1)]
void main(
    uint3 global_id : SV_DispatchThreadID,
    uint3 t_id : SV_GroupThreadID
)
{
	uint global_id = global_id.x;
	uint count = pc.count;
	if (global_id >= count)
		return;

	uint id = pc.mesh_indices_buffer[global_id];

	ObjectData obj = pc.object_buffer[id];
	uint mesh_id = obj.mesh_id;
	MeshData mesh = pc.mesh_buffer[mesh_id];

	bool alpha_clip = obj.post_pass != 0;

	float radius = mesh.radius;

	// non uniform scaling not supported
	float max_scale = obj.scale;
	radius = radius * max_scale;

	uint cascade_index = t_id.y;

    float cascade_radius = uniforms.shadow_widths[cascade_index];

    float3 mpos = mesh.center;
    float3 center = rotate_quat(mpos, obj.orientation) * obj.scale + obj.translation; // world space

    if (!partially_inside_cascade(center, radius, cascade_index))
        return;

    // https://learn.microsoft.com/en-us/windows/win32/dxtecharts/cascaded-shadow-maps
    // incompatible with interval-based map selection?
    //if (map == 1 && cull == 1)
    //{
    //    if (cascade_index > 0 && completely_inside_cascade(center, radius, 0))
    //        return;
    //
    //    if (cascade_index > 1 && completely_inside_cascade(center, radius, 1))
    //        return;
    //
    //    if (cascade_index > 2 && completely_inside_cascade(center, radius, 2))
    //        return;
    //}

    uint lod_index = 0;
    if (pc.lod_enabled == 1)
    {
        float d = max(0.0, length(center) - radius);

        // for orthographic, p11 = 2 / (top - bottom)
        // lod_distance_factor = 2 / (p11 * height);
        // we can factor out both the 2s, then top-bottom == 2*top
        float lod_distance_factor = 2 * cascade_radius / 4096.0;
        float e = lod_distance_factor / max_scale;

        for (uint i = 0; i < mesh.lod_count; i++)
        {
            lod_index = e >= mesh.lods[i].error ? i : lod_index;
        }
    }

    alpha_clip ? InterlockedAdd(pc.draw_buffer.draws[cascade_index].alpha_clip_count, 1, count) : InterlockedAdd(pc.draw_buffer.draws[cascade_index].opaque_count, 1, count);
    uint index = alpha_clip ? count + 200000 : count;

    DrawCommand draw;
	draw.index_count = mesh.lods[lod_index].count;
	draw.instance_count = 1;
	draw.first_index = mesh.lods[lod_index].first_index;
	draw.vertex_offset = mesh.vertex_offset;
	draw.first_instance = id;

    pc.draw_buffer.draws[cascade_index].commands[index] = draw;
}
