#version 460

#extension GL_GOOGLE_include_directive : require

#include "bindings.glsl"
#include "buffer_references.glsl"

layout (location = 0) flat in uint in_draw_id;
layout (location = 1) flat in uint in_triangle_id;
layout (location = 2) in vec2 in_uv;
layout (location = 3) flat in uint in_material_id;
layout (location = 4) in vec4 in_clip_pos;
layout (location = 5) in vec4 in_prev_clip_pos;

layout (location = 0) out uvec2 out_visibility_id;
layout (location = 1) out vec2 out_velocity;

// 1 - OPAQUE
// 0 - MASK
layout (constant_id = 0) const int OPAQUE = 1;

layout( push_constant ) uniform constants
{
	ObjectBuffer object_buffer;
	VertexBuffer vertex_buffer;
	MeshletBuffer meshlet_buffer;
	MeshletIndicesBuffer meshlet_indices_buffer;
	ClusterIndicesBuffer cluster_indices_buffer;
	MaterialBuffer material_buffer;
	OITBuffer oit_buffer;
	uint padding[2];
	vec2 jitter_offset;
};

void main()
{
	if (OPAQUE == 0)
	{
		MaterialData m = material_buffer.materials[in_material_id];
		vec4 albedo = texture(sampler2D(textures[m.diffuse_id], samplers[LINEAR_SAMPLER]), in_uv);

		if (albedo.a < 0.5)
			discard;

	}

	out_visibility_id = uvec2(in_draw_id, in_triangle_id);

	// TODO: leave camera motion here for now until we get things working, then move it out
    vec2 curr_ndc = in_clip_pos.xy / in_clip_pos.w;
    vec2 prev_ndc = in_prev_clip_pos.xy / in_prev_clip_pos.w;
    vec2 velocity = curr_ndc - prev_ndc;

    // TODO:
    velocity = velocity * vec2(0.5, -0.5) + 0.5;

    // TODO: bake cpu side possible?
    // note: take current and previous jitter into account!
    velocity -= jitter_offset;

    out_velocity = velocity;
}
