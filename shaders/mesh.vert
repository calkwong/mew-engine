#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "scene.glsl"
#include "mesh.glsl"

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outUV;
layout (location = 2) out vec4 outTangent;
layout (location = 3) out flat uint outMaterialID;

layout(buffer_reference, std430) readonly buffer VertexBuffer
{ 
	Vertex vertices[];
};

layout(buffer_reference, std430) readonly buffer ObjectBuffer
{ 
	ObjectData objects[];
};

layout( push_constant ) uniform constants
{
	ObjectBuffer objectBuffer;
	VertexBuffer vertexBuffer;
	//MeshTaskBuffer meshTaskBuffer;
	//MeshletBuffer meshletBuffer;
	//MeshletIndicesBuffer meshletIndicesBuffer;
	//ClusterIndicesBuffer clusterIndicesBuffer; 
	//MaterialBuffer materialBuffer;
	//OITBuffer oitBuffer;
	uint padding[6 * 2];
	uint debugMeshlets;
} pc;

void main() 
{
	ObjectData o = pc.objectBuffer.objects[gl_InstanceIndex]; // gl_InstanceIndex from drawIndirectCommand
	Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
	
	vec4 position = o.worldMatrix * vec4(v.position, 1.0);

	//outNormal = mat3(transpose(inverse(o.worldMatrix))) * v.normal;
	//outTangent = vec4(mat3(transpose(inverse(o.worldMatrix))) * v.tangent.xyz, v.tangent.w);
	
	outNormal = mat3(o.worldMatrix) * v.normal;
	outTangent = vec4(mat3(o.worldMatrix) * v.tangent.xyz, v.tangent.w);
	
	outUV = vec2(v.uv_x, v.uv_y);
	outMaterialID = o.materialID;
	
	gl_Position =  sceneData.viewproj * position;
}
