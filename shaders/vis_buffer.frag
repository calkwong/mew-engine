#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene.glsl"
#include "mesh.glsl"
#include "samplers.glsl"

layout (location = 0) flat in uint inDrawID;
layout (location = 1) flat in uint inTriangleID;
layout (location = 2) in vec2 inUV;
layout (location = 3) flat in uint inMaterialID;
layout (location = 4) in vec4 inClipPos;
layout (location = 5) in vec4 inPrevClipPos;

//layout (location = 0) out uvec4 outFragColor;
layout (location = 0) out uvec2 visibilityID;
layout (location = 1) out vec2 velocity;

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
	uint padding3;
	vec2 jitterOffset;
} pc;

layout(set = 1, binding = 0) uniform texture2D allTextures[];
layout(set = 1, binding = 0) uniform textureCube allCubemaps[];
layout(set = 2, binding = 0) uniform sampler samplers[];

void main() 
{	
	
	if (OPAQUE == 0)
	{
		MaterialData m = pc.materialBuffer.materials[inMaterialID];
		vec4 albedo = texture(sampler2D(allTextures[m.diffuseID], samplers[LINEAR_SAMPLER]), inUV);
		
		if (albedo.a < 0.5)
			discard;
		
	}

	//outFragColor = uvec4(inDrawID, inTriangleID, 0, 0);
	visibilityID = uvec2(inDrawID, inTriangleID);
	
	vec2 currentNdc = inClipPos.xy / inClipPos.w;
	vec2 previousNdc = inPrevClipPos.xy / inPrevClipPos.w;

	vec2 velocity = currentNdc - previousNdc;
	velocity = velocity * 0.5 + 0.5; 
	velocity.y *= -1.0; // flip for uv space
	
	velocity -= pc.jitterOffset;
}