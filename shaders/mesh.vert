#version 460

#extension GL_GOOGLE_include_directive : require

#include "bindings.glsl"
#include "buffer_references.glsl"
#include "util.glsl"

layout (location = 0) out vec3 out_normal;
layout (location = 1) out vec2 out_uv;
layout (location = 2) out vec4 out_tangent;
layout (location = 3) out flat uint out_material_id;
layout (location = 4) out vec4 out_clip_pos;
layout (location = 5) out vec4 out_prev_clip_pos;

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
	ObjectData o = object_buffer.objects[gl_InstanceIndex]; // gl_InstanceIndex from drawIndirectCommand
	Vertex v = vertex_buffer.vertices[gl_VertexIndex];

	vec3 pos = vec3(v.px, v.py, v.pz);
	vec4 world_pos = vec4(rotate_quat(pos, o.orientation) * o.scale + o.translation, 1.0);

	vec3 normal;
	vec4 tangent;
	unpack_tbn(v.normal, uint(v.tangent), normal, tangent);

	out_normal = rotate_quat(normal, o.orientation);
	out_tangent = vec4(rotate_quat(tangent.xyz, o.orientation), tangent.w);

	out_uv = vec2(v.uv_x, v.uv_y);
	out_material_id = o.material_id;

    vec4 clip_pos = uniforms.view_proj * world_pos;
    vec4 prev_clip_pos = uniforms.prev_view_proj * world_pos;

    out_clip_pos = clip_pos;
    out_prev_clip_pos = prev_clip_pos;

	gl_Position = clip_pos;
}
