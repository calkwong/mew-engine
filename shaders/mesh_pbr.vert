#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "scene.glsl"

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outWorldPos;
layout (location = 2) out vec2 outUV;
layout (location = 3) out vec4 outTangent;

struct Vertex {
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 tangent;
}; 

struct MaterialData
{
	vec4 baseColorFactor;
	float metallicFactor;
	float roughnessFactor;
	uint diffuseID;
	uint metalRoughnessID;
	uint normalID;
	uint occlusionID;
	uint emissiveID;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer
{ 
	Vertex vertices[];
};

layout(buffer_reference, std430) readonly buffer MaterialBuffer
{ 
	MaterialData materials[];
};

layout(push_constant) uniform constants
{
	mat4 worldMatrix;
	VertexBuffer vertexBuffer;
	MaterialBuffer materialBuffer;
	uint materialID;
} pc;

void main() 
{
	Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
	
	vec4 position = pc.worldMatrix * vec4(v.position, 1.0);
	gl_Position =  sceneData.viewproj * position;

	outNormal = mat3(transpose(inverse(pc.worldMatrix))) * v.normal;
	outTangent = vec4(mat3(transpose(inverse(pc.worldMatrix))) * v.tangent.xyz, v.tangent.w);
	outWorldPos = position.xyz;
	
	outUV = vec2(v.uv_x, v.uv_y);
}
