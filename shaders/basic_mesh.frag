#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene.glsl"
#include "mesh.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inWorldPos;
layout (location = 2) in vec3 inViewPos;
layout (location = 3) in vec2 inUV;
layout (location = 4) in vec4 inTangent;
layout (location = 5) in flat uint inMaterialID;

layout (location = 0) out vec4 outFragColor;

const float exposure = 4.0;
const float gamma = 2.2;
const float PI = 3.14159265359;

layout( push_constant ) uniform constants
{
	//ObjectBuffer objectBuffer;
	//VertexBuffer vertexBuffer;
	//MeshBuffer meshBuffer;
	//MeshTaskBuffer meshTaskBuffer;
	//MeshletBuffer meshletBuffer;
	//MeshletIndicesBuffer meshletIndicesBuffer;
	//CountBuffer countBuffer;
	uint padding[14];
	uint debugMeshlets;
} pc;

layout(set = 1, binding = 0) uniform texture2D allTextures[];
layout(set = 1, binding = 0) uniform textureCube allCubemaps[];
layout(set = 2, binding = 0) uniform sampler samplers[];

void main() 
{	
	vec3 N = normalize(inNormal);
	outFragColor = vec4(N, 1);
	outFragColor.xyz = outFragColor.xyz * 0.5 + 0.5;
	
	if (pc.debugMeshlets == 1)
		outFragColor = vec4(N, 1);
	
}