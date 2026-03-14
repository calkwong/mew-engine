#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_mesh_shader : require

#include "bindings.glsl"
#include "buffer_references.glsl"
#include "util.glsl"

layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

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
	vec2 jitterOffset;
} pc;

layout (location = 0) out vec3 outNormal[];
layout (location = 1) out vec2 outUV[];
layout (location = 2) out vec4 outTangent[];
layout (location = 3) out flat uint outMaterialID[];
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
		uint vertexIndex = pc.meshletIndicesBuffer.indices[m_vertexOffset + i];
		Vertex v = pc.vertexBuffer.vertices[vertexIndex];

		vec4 position = o.worldMatrix * vec4(v.position, 1.0);
		
		vec3 normal;
		vec4 tangent;
		unpackTBN(v.normal, uint(v.tangent), normal, tangent);
		
		outNormal[i] = mat3(o.worldMatrix) * normal;
		outTangent[i] = vec4(mat3(o.worldMatrix) * tangent.xyz, tangent.w);
		
		//outNormal[i] = mat3(transpose(inverse(o.worldMatrix))) * v.normal;
		//outTangent[i] = vec4(mat3(transpose(inverse(o.worldMatrix))) * v.tangent.xyz, v.tangent.w);
		
		if (pc.debugMeshlets == 1)
		{
			uint mhash = hash(meshletOffset);
			outNormal[i] = vec3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
		}
		
		outUV[i] = vec2(v.uv_x, v.uv_y);
		outMaterialID[i] = o.materialID;

	    vec4 clipPos = sceneData.viewproj * position;
        vec4 prevClipPos = sceneData.previousViewproj * position;

        outClipPos[i] = clipPos;
	    outPrevClipPos[i] = prevClipPos;

		gl_MeshVerticesEXT[i].gl_Position = sceneData.viewproj * position;
	}
	
	//barrier();
	//memoryBarrier();
	
	for (uint i = ti; i < triangleCount; i += 32)
	{
		uint idx0 = pc.meshletIndicesBuffer.indices[m_triangleOffset + i * 3 + 0];
		uint idx1 = pc.meshletIndicesBuffer.indices[m_triangleOffset + i * 3 + 1];
		uint idx2 = pc.meshletIndicesBuffer.indices[m_triangleOffset + i * 3 + 2];
	
		gl_PrimitiveTriangleIndicesEXT[i] =  uvec3(idx0, idx1, idx2);
	}
}