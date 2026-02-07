#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "mesh.glsl"
#include "scene.glsl"

layout(set = 1, binding = 0) uniform texture2D allTextures[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout (buffer_reference, std430) readonly buffer VertexBuffer 
{
	Vertex vertices[];
};

layout(buffer_reference, std430) readonly buffer ObjectBuffer
{ 
	ObjectData objects[];
};

layout (push_constant) uniform constants
{
	mat4 viewproj;
	//MaterialBuffer materialBuffer;
	uint padding[1 * 2];
	ObjectBuffer objectBuffer;
	VertexBuffer vertexBuffer;
} pc;

layout (location = 0) out vec2 outUV;
layout (location = 1) flat out uint outMaterialID;

void main()
{
	ObjectData o = pc.objectBuffer.objects[gl_InstanceIndex]; // gl_InstanceIndex from drawIndirectCommand
	Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
	
	vec4 position = o.worldMatrix * vec4(v.position, 1.0);
	gl_Position = pc.viewproj * position;
	
	outUV = vec2(v.uv_x, v.uv_y);
	outMaterialID = o.materialID;
}