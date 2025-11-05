#version 450
#extension GL_EXT_buffer_reference : require

struct Vertex
{
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 tangent;
};

layout (buffer_reference, std430) readonly buffer VertexBuffer {
	Vertex vertices[];
};

// push constant block
layout (push_constant) uniform constants
{
	mat4 world_matrix;
	mat4 viewproj;
	VertexBuffer vertexBuffer;
} pc;

layout (location = 0) out vec2 outUV;

void main()
{
	Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
	outUV = vec2(v.uv_x, v.uv_y); // is this wasted?
	gl_Position = pc.viewproj * pc.world_matrix * vec4(v.position, 1.0);
}