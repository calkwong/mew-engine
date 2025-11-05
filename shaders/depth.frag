#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene.glsl"

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

layout(buffer_reference, std430) readonly buffer MaterialBuffer
{ 
	MaterialData materials[];
};

// push constant block
layout (push_constant) uniform constants
{
	mat4 world_matrix;
	mat4 viewproj;
	VertexBuffer vertexBuffer;
	MaterialBuffer materialData;
	uint materialID;
} pc;

layout (location = 0) in vec2 inUV;

layout(set = 1, binding = 0) uniform texture2D allTextures[];
layout(set = 2, binding = 0) uniform sampler samplers[];

void main()
{
	MaterialData m = pc.materialData.materials[pc.materialID];
		
	vec4 albedo = m.baseColorFactor;
	if (m.diffuseID != 0)
		albedo *= texture(sampler2D(allTextures[m.diffuseID], samplers[0]), inUV);
		
	if (albedo.a < 0.5)
		discard;
}