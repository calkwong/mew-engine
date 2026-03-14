#version 460

#extension GL_GOOGLE_include_directive : require

#include "bindings.glsl"
#include "buffer_references.glsl"
#include "util.glsl"

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outUV;
layout (location = 2) out vec4 outTangent;
layout (location = 3) out flat uint outMaterialID;
layout (location = 4) out vec4 outClipPos;
layout (location = 5) out vec4 outPrevClipPos;

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
	vec2 jitterOffset;
} pc;

void main() 
{
	ObjectData o = pc.objectBuffer.objects[gl_InstanceIndex]; // gl_InstanceIndex from drawIndirectCommand
	Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
	
	vec4 position = o.worldMatrix * vec4(v.position, 1.0);

	vec3 normal;
	vec4 tangent;
	unpackTBN(v.normal, uint(v.tangent), normal, tangent);
	
	outNormal = mat3(o.worldMatrix) * normal;
	outTangent = vec4(mat3(o.worldMatrix) * tangent.xyz, tangent.w);
	
	//outNormal = mat3(transpose(inverse(o.worldMatrix))) * v.normal;
	//outTangent = vec4(mat3(transpose(inverse(o.worldMatrix))) * v.tangent.xyz, v.tangent.w);
	
	outUV = vec2(v.uv_x, v.uv_y);
	outMaterialID = o.materialID;

    outClipPos = sceneData.viewproj * position;
    outPrevClipPos = sceneData.previousViewproj * position;

	gl_Position =  sceneData.viewproj * position;
}
