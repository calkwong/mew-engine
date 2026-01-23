#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene.glsl"
#include "mesh.glsl"
#include "samplers.glsl"

// TODO: clean up - lots of redundant interpolants based on old pbr code that's been put aside
layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inWorldPos;
layout (location = 2) in vec3 inViewPos;
layout (location = 3) in vec2 inUV;
layout (location = 4) in vec4 inTangent;
layout (location = 5) in flat uint inMaterialID;

//layout (location = 0) out vec4 outFragColor;
layout (location = 0) out vec4 gbuffer[2];

const float exposure = 4.0;
const float gamma = 2.2;
const float PI = 3.14159265359;

layout(buffer_reference, std430) readonly buffer MaterialBuffer
{ 
	MaterialData materials[];
};

layout( push_constant ) uniform constants
{
	//ObjectBuffer objectBuffer;
	//VertexBuffer vertexBuffer;
	//MeshBuffer meshBuffer;
	//MeshTaskBuffer meshTaskBuffer;
	//MeshletBuffer meshletBuffer;
	//MeshletIndicesBuffer meshletIndicesBuffer;
	//CountBuffer countBuffer;
	//ClusterIndicesBuffer clusterIndicesBuffer; 
	uint padding[16];
	MaterialBuffer materialBuffer;
	uint debugMeshlets;
} pc;

layout(set = 1, binding = 0) uniform texture2D allTextures[];
layout(set = 1, binding = 0) uniform textureCube allCubemaps[];
layout(set = 2, binding = 0) uniform sampler samplers[];

void main() 
{	
	MaterialData m = pc.materialBuffer.materials[inMaterialID];
	
	vec4 albedo = m.baseColorFactor;
	if (m.diffuseID != 0)
		albedo *= texture(sampler2D(allTextures[m.diffuseID], samplers[0]), inUV);

	vec3 N = normalize(inNormal); 
	
	if (pc.debugMeshlets == 0)
	{
		//outFragColor = albedo;
		N = N.xyz * 0.5 + 0.5;
		//outFragColor = vec4(N, 1);
		
		gbuffer[0] = albedo;
		gbuffer[1] = vec4(N, 1);
	}
	else // visualize meshlets
	{
		//outFragColor = vec4(N, 1); 
		
		gbuffer[0] = albedo;
		gbuffer[1] = vec4(N, 1);
	}
	
	
	
}