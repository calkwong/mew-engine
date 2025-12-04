#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "scene.glsl"
#include "mesh.glsl"

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outWorldPos;
layout (location = 2) out vec3 outViewPos;
layout (location = 3) out vec2 outUV;
layout (location = 4) out vec4 outTangent;
layout (location = 5) flat out uint outMaterialID;

layout(buffer_reference, std430) readonly buffer VertexBuffer
{ 
	Vertex vertices[];
};

layout(buffer_reference, std430) readonly buffer MaterialBuffer
{ 
	MaterialData materials[];
};

layout(buffer_reference, std430) readonly buffer ObjectBuffer
{ 
	ObjectData objects[];
};

layout(buffer_reference, std430) readonly buffer InstanceBuffer
{ 
	uint instances[];
};

layout( push_constant ) uniform constants
{
	ObjectBuffer objectBuffer;
	VertexBuffer vertexBuffer;
	//MeshBuffer meshBuffer;
	//MeshTaskBuffer meshTaskBuffer;
	//MeshletBuffer meshletBuffer;
	//MeshletIndicesBuffer meshletIndicesBuffer;
	//CountBuffer countBuffer;
	//ClusterIndicesBuffer clusterIndicesBuffer; 
	uint padding[12]; 
	uint debugMeshlets;
} pc;

void main() 
{
	ObjectData o = pc.objectBuffer.objects[gl_InstanceIndex];
	Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
	
	vec4 position = o.worldMatrix * vec4(v.position, 1.0);

	outNormal = mat3(transpose(inverse(o.worldMatrix))) * v.normal;
	
	outViewPos = vec3(sceneData.view * position);
	outTangent = vec4(mat3(transpose(inverse(o.worldMatrix))) * v.tangent.xyz, v.tangent.w);
	outWorldPos = position.xyz;
	
	outUV = vec2(v.uv_x, v.uv_y);
	outMaterialID = o.materialID;
	
	gl_Position =  sceneData.viewproj * position;
}
