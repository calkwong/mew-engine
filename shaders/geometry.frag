#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene.glsl"
#include "mesh.glsl"
#include "samplers.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec4 inTangent;
layout (location = 3) in flat uint inMaterialID;

layout (location = 0) out vec4 gbuffer[];

// 1 - OPAQUE
// 0 - MASK
layout (constant_id = 0) const int OPAQUE = 1;


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

#define PBR

void main() 
{	
	MaterialData m = pc.materialBuffer.materials[inMaterialID];
	
	vec4 albedo = m.baseColorFactor;
	if (m.diffuseID != 0)
	{
		vec4 sampledAlbedo = texture(sampler2D(allTextures[m.diffuseID], samplers[LINEAR_SAMPLER]), inUV);
		
		if (OPAQUE == 0)
		{
			if (sampledAlbedo.a < 0.5)
				discard;
		}
		
		albedo *= sampledAlbedo;
	}
	
	vec3 N;
	float metallic;
	float roughness;
#ifdef PBR
	N = normalize(inNormal); // mikktspace convention is NOT to normalize? but khronos sponza breaks	
	if (m.normalID != 0)
	{
		vec3 T = normalize(inTangent.xyz); // mikktspace convention is NOT to normalize? but khronos sponza breaks	
		float sign = inTangent.w; // sign is flipped during tangent generation so mikktspace is consistent with glTF handedness
		vec3 B = sign * cross(N, T);
	
		vec3 shadingNormal = texture(sampler2D(allTextures[m.normalID], samplers[LINEAR_SAMPLER]), inUV).xyz;
		shadingNormal = shadingNormal * 2.0 - 1.0;
		N = normalize(shadingNormal.x * T.xyz + shadingNormal.y * B + shadingNormal.z * N);
	}
	
	metallic = m.metallicFactor;
	float perceptualRoughness = m.roughnessFactor;
	vec2 metalRoughness = vec2(0.0);
	if (m.metalRoughnessID != 0)
	{
		metalRoughness = texture(sampler2D(allTextures[m.metalRoughnessID], samplers[LINEAR_SAMPLER]), inUV).bg;
		metallic *= metalRoughness.x;
		perceptualRoughness *= metalRoughness.y;
	}
	perceptualRoughness = max(perceptualRoughness, 0.045); // frostbite engine clamp value for analytical lights (fp32)
	roughness = perceptualRoughness;
	//roughness = perceptualRoughness * perceptualRoughness; // we could maybe do this during lighting so we maintain perceptual roughness debuggability
#else
	N = normalize(inNormal); 
#endif
	
	if (pc.debugMeshlets == 0)
	{
		//N = N.xyz * 0.5 + 0.5;
		
		gbuffer[0] = albedo;
		gbuffer[1] = vec4(N, 1);
		gbuffer[2] = vec4(metallic, roughness, 0.0, 1.0);
	}
	else // visualize meshlets
	{
		gbuffer[0] = albedo; // don't matter
		gbuffer[1] = vec4(N, 1); // meshlet color
		gbuffer[2] = vec4(metallic, roughness, 0.0, 1.0); // don't matter
	}
}