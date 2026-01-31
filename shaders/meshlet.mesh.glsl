#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_mesh_shader : require

#include "scene.glsl"
#include "mesh.glsl"
#include "samplers.glsl"

layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

//out uvec3 gl_PrimitiveTriangleIndicesEXT[];

layout(buffer_reference, std430) readonly buffer VertexBuffer
{ 
	Vertex vertices[];
};

layout(buffer_reference, std430) readonly buffer ObjectBuffer
{ 
	ObjectData objects[];
};

layout(buffer_reference, std430) buffer MeshTaskBuffer
{ 
	MeshTaskCommand commands[];
};

layout(buffer_reference, std430) readonly buffer MeshBuffer
{ 
	MeshData meshes[];
};

layout(buffer_reference, std430) readonly buffer MeshletBuffer
{ 
	Meshlet meshlets[];
};

layout(buffer_reference, std430) readonly buffer MeshletIndicesBuffer
{ 
	uint indices[];
};

layout(buffer_reference, std430) readonly buffer MaterialBuffer
{ 
	MaterialData materials[];
};

layout(buffer_reference, std430) buffer CountBuffer
{
	uint count;
	uint workgroupX;
	uint workgroupY;
	uint workgroupZ;
};

layout(buffer_reference, std430) buffer ClusterIndicesBuffer
{ 
	uint indices[];
};

layout( push_constant ) uniform constants
{
	ObjectBuffer objectBuffer;
	VertexBuffer vertexBuffer;
	MeshTaskBuffer meshTaskBuffer;
	MeshletBuffer meshletBuffer;
	MeshletIndicesBuffer meshletIndicesBuffer;
	ClusterIndicesBuffer clusterIndicesBuffer; 
	//MaterialBuffer materialBuffer;
	uint padding[1 * 2];
	uint debugMeshlets;
} pc;

layout (location = 0) out vec3 outNormal[];
layout (location = 1) out vec3 outWorldPos[];
layout (location = 2) out vec3 outViewPos[];
layout (location = 3) out vec2 outUV[];
layout (location = 4) out vec4 outTangent[];
layout (location = 5) flat out uint outMaterialID[];

uint hash(uint a)
{
   a = (a+0x7ed55d16) + (a<<12);
   a = (a^0xc761c23c) ^ (a>>19);
   a = (a+0x165667b1) + (a<<5);
   a = (a+0xd3a2646c) ^ (a<<9);
   a = (a+0xfd7046c5) + (a<<3);
   a = (a^0xb55a4f09) ^ (a>>16);
   return a;
}

void main() 
{
	
	uint id = gl_WorkGroupID.x;
	
	uint value = pc.clusterIndicesBuffer.indices[id];
	
	// extract bits
	uint offset = bitfieldExtract(value, 27, 5);
	uint meshTaskId = bitfieldExtract(value, 0, 27);
		
	MeshTaskCommand command = pc.meshTaskBuffer.commands[meshTaskId];
	uint meshletOffset = command.meshletOffset + offset;
	Meshlet meshlet = pc.meshletBuffer.meshlets[meshletOffset];
	uint vertexCount = meshlet.vertexCount;
	uint triangleCount = meshlet.triangleCount;
	
	SetMeshOutputsEXT(vertexCount, triangleCount);

	uint objectId = command.objectId;
	ObjectData o = pc.objectBuffer.objects[objectId];
	
	uint m_vertexOffset = meshlet.dataOffset; // offset into meshletIndicesBuffer
	uint m_triangleOffset = m_vertexOffset + vertexCount;
	
	uint ti = gl_LocalInvocationIndex;
	
	for (uint i = ti; i < vertexCount; i += 32)
	{
		uint vertexIndex = pc.meshletIndicesBuffer.indices[m_vertexOffset + i];
		Vertex v = pc.vertexBuffer.vertices[vertexIndex];

		vec4 position = o.worldMatrix * vec4(v.position, 1.0);
		outNormal[i] = mat3(transpose(inverse(o.worldMatrix))) * v.normal;
		
		if (pc.debugMeshlets == 1)
		{
			uint mhash = hash(meshletOffset);
			outNormal[i] = vec3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
		}
		
		outViewPos[i] = vec3(sceneData.view * position);
		outTangent[i] = vec4(mat3(transpose(inverse(o.worldMatrix))) * v.tangent.xyz, v.tangent.w);
		outWorldPos[i] = position.xyz;
		outUV[i] = vec2(v.uv_x, v.uv_y);
		outMaterialID[i] = o.materialID;
	
		gl_MeshVerticesEXT[i].gl_Position = sceneData.viewproj * position;
	}
	
	memoryBarrier();
	barrier();
	
	for (uint i = ti; i < triangleCount; i += 32)
	{
		uint idx0 = pc.meshletIndicesBuffer.indices[m_triangleOffset + i * 3 + 0];
		uint idx1 = pc.meshletIndicesBuffer.indices[m_triangleOffset + i * 3 + 1];
		uint idx2 = pc.meshletIndicesBuffer.indices[m_triangleOffset + i * 3 + 2];
	
		gl_PrimitiveTriangleIndicesEXT[i] =  uvec3(idx0, idx1, idx2);
	}
}