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

layout (location = 0) out vec4 gbuffer[];

// 1 - OPAQUE
// 0 - MASK
layout (constant_id = 0) const int OPAQUE = 1;

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
	//MeshTaskBuffer meshTaskBuffer;
	//MeshletBuffer meshletBuffer;
	//MeshletIndicesBuffer meshletIndicesBuffer;
	//ClusterIndicesBuffer clusterIndicesBuffer; 
	uint padding[6 * 2];
	MaterialBuffer materialBuffer;
	//OITBuffer oitBuffer;
	uint padding2[1 * 2];
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
	{
		vec4 sampledAlbedo = texture(sampler2D(allTextures[m.diffuseID], samplers[0]), inUV);
		
		if (OPAQUE == 0)
		{
			if (sampledAlbedo.a < 0.5)
				discard;
		}
		
		albedo *= sampledAlbedo;
	}
	vec3 N = normalize(inNormal); 
	
	if (pc.debugMeshlets == 0)
	{
		//outFragColor = albedo;
		N = N.xyz * 0.5 + 0.5;
		//outFragColor = vec4(N, 1);
		
		gbuffer[0] = albedo;
		gbuffer[1] = vec4(N, 1);
		//gbuffer[2] = vec4(inViewPos, 1.0);
		gbuffer[2] = vec4(inWorldPos, 1.0);
	}
	else // visualize meshlets
	{
		//outFragColor = vec4(N, 1); 
		
		gbuffer[0] = albedo;
		gbuffer[1] = vec4(N, 1);
		//gbuffer[2] = vec4(inViewPos, 1);
		gbuffer[2] = vec4(inWorldPos, 1.0);
	}
	
	
	
}