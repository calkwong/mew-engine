// TODO: REFACTOR, PLENTY OF COMMITS BEHIND

#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene.glsl"

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
	mat4 viewproj;
	MaterialBuffer materialBuffer;
//	ObjectBuffer objectBuffer;
//	VertexBuffer vertexBuffer;
} pc;

layout (location = 0) in vec2 inUV;
layout (location = 1) flat in uint inMaterialID;

layout(set = 1, binding = 0) uniform texture2D allTextures[];
layout(set = 2, binding = 0) uniform sampler samplers[];

void main()
{
	MaterialData m = pc.materialBuffer.materials[inMaterialID];
		
	vec4 albedo = m.baseColorFactor;
	if (m.diffuseID != 0)
		albedo *= texture(sampler2D(allTextures[m.diffuseID], samplers[0]), inUV);
		
	if (albedo.a < 0.5)
		discard;
}