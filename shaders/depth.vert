#version 450

#extension GL_GOOGLE_include_directive : require

#include "bindings.glsl"
#include "buffer_references.glsl"
#include "util.glsl"

layout (push_constant) uniform constants
{
	mat4 view_proj;
	MaterialBuffer material_buffer;
	ObjectBuffer object_buffer;
	VertexBuffer vertex_buffer;
};

layout (location = 0) out vec2 out_uv;
layout (location = 1) flat out uint out_material_id;

void main()
{
	ObjectData o = object_buffer.objects[gl_InstanceIndex]; // gl_InstanceIndex from drawIndirectCommand
	Vertex v = vertex_buffer.vertices[gl_VertexIndex];
	
	vec3 pos = vec3(v.px, v.py, v.pz);
	vec4 world_pos = vec4(rotate_quat(pos, o.orientation) * o.scale + o.translation, 1.0);
	
	gl_Position = view_proj * world_pos;
	
	out_uv = vec2(v.uv_x, v.uv_y);
	out_material_id = o.material_id;
}