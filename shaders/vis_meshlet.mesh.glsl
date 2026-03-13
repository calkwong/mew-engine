#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_mesh_shader : require

#include "bindings.glsl"
#include "buffer_references.glsl"

layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out; // not 126?

layout( push_constant ) uniform constants
{
	ObjectBuffer objectBuffer;
	VertexBuffer vertexBuffer;
	MeshTaskBuffer meshTaskBuffer;
	MeshletBuffer meshletBuffer;
	MeshletIndicesBuffer meshletIndicesBuffer;
	ClusterIndicesBuffer clusterIndicesBuffer; 
	//MaterialBuffer materialBuffer;
	//OITBuffer oitBuffer;
	uint padding[2 * 2];
	uint debugMeshlets;
	uint padding2;
	vec2 jitterOffset;
} pc;

layout (location = 0) perprimitiveEXT out uint outDrawID[];
layout (location = 1) perprimitiveEXT out uint outTriangleID[];
layout (location = 2) out vec2 outUV[];
layout (location = 3) out uint outMaterialID[];
layout (location = 4) out vec4 outClipPos[];
layout (location = 5) out vec4 outPrevClipPos[];

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
		uint vertexIndex = pc.meshletIndicesBuffer.indices[m_vertexOffset + i]; // vertex indices for meshlets are global, really need to document this
		Vertex v = pc.vertexBuffer.vertices[vertexIndex];
		vec4 position = o.worldMatrix * vec4(v.position, 1.0);
	
		vec4 clipPos = sceneData.viewproj * position;
		vec4 prevClipPos = sceneData.previousViewproj * position;
		
		outUV[i] = vec2(v.uv_x, v.uv_y);
		outMaterialID[i] = o.materialID; // only used for alpha clipping
		
		outClipPos[i] = clipPos;
		outPrevClipPos[i] = prevClipPos;
		
		gl_MeshVerticesEXT[i].gl_Position = clipPos;
	}
	
	//barrier();
	//memoryBarrier(); 
	
	for (uint i = ti; i < triangleCount; i += 32)
	{
		outDrawID[i] = objectId;
		
		// packing meshletID and triangleID
		outTriangleID[i] = (i << 25) | meshletOffset; // no error checking yet
		
		uint idx0 = pc.meshletIndicesBuffer.indices[m_triangleOffset + i * 3 + 0];
		uint idx1 = pc.meshletIndicesBuffer.indices[m_triangleOffset + i * 3 + 1];
		uint idx2 = pc.meshletIndicesBuffer.indices[m_triangleOffset + i * 3 + 2];
	
		gl_PrimitiveTriangleIndicesEXT[i] =  uvec3(idx0, idx1, idx2);
	}
}