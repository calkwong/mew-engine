#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "bindings.glsl"
#include "buffer_references.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec4 inTangent;
layout (location = 3) in flat uint inMaterialID;
layout (location = 4) in vec4 inClipPos;
layout (location = 5) in vec4 inPrevClipPos;

layout (location = 0) out vec4 outAlbedo;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec2 outMetalRoughness;
layout (location = 3) out vec2 outVelocity;

// 1 - OPAQUE
// 0 - MASK
layout (constant_id = 0) const int OPAQUE = 1;

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
	vec2 jitterOffset;
} pc;

#define PBR

void main() 
{	
	MaterialData m = pc.materialBuffer.materials[inMaterialID];
	
	vec4 albedo = m.baseColorFactor;
	if (m.diffuseID != 0)
	{
		vec4 sampledAlbedo = texture(sampler2D(textures[m.diffuseID], samplers[LINEAR_SAMPLER]), inUV);
		
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
	//N = inNormal; 
	N = normalize(inNormal); // TODO: mikktspace convention is NOT to normalize. we normalize here as khronos sponza breaks iirc?
	
	if (OPAQUE == 0)
	{
		N = gl_FrontFacing ? N : -N;
	}
	
	if (m.normalID != 0)
	{
		vec3 T = normalize(inTangent.xyz); // TODO: mikktspace convention is NOT to normalize. we normalize here as khronos sponza breaks iirc?
		//vec3 T = inTangent.xyz;
		float sign = inTangent.w; // sign is flipped during tangent generation so mikktspace is consistent with glTF handedness
		
		if (OPAQUE == 0)
			sign = gl_FrontFacing ? sign : -sign;
		
		vec3 B = sign * cross(N, T);
		
		vec3 shadingNormal = texture(sampler2D(textures[m.normalID], samplers[LINEAR_SAMPLER]), inUV).xyz;
		shadingNormal = shadingNormal * 2.0 - 1.0;
		N = normalize(shadingNormal.x * T.xyz + shadingNormal.y * B + shadingNormal.z * N);
	}
	
	metallic = m.metallicFactor;
	float perceptualRoughness = m.roughnessFactor;
	vec2 metalRoughness = vec2(0.0);
	if (m.metalRoughnessID != 0)
	{
		metalRoughness = texture(sampler2D(textures[m.metalRoughnessID], samplers[LINEAR_SAMPLER]), inUV).bg;
		metallic *= metalRoughness.x;
		perceptualRoughness *= metalRoughness.y;
	}
	perceptualRoughness = max(perceptualRoughness, 0.045); // frostbite engine clamp value for analytical lights (fp32)
#else
	N = normalize(inNormal); 
#endif

    vec2 currentNdc = inClipPos.xy / inClipPos.w;
    vec2 previousNdc = inPrevClipPos.xy / inPrevClipPos.w;

    vec2 velocity = currentNdc - previousNdc;
    velocity = velocity * 0.5 + 0.5;
    velocity.y *= -1.0; // flip for velocity in uv space

    velocity -= pc.jitterOffset;

    outAlbedo = albedo;
    outNormal = vec4(N, 1);
    outMetalRoughness = vec2(metallic, perceptualRoughness);
    outVelocity = vec2(velocity);
}