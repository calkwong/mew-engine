#version 450

#extension GL_GOOGLE_include_directive : require

#include "bindings.glsl"
#include "buffer_references.glsl"
#include "util.glsl"

layout (push_constant) uniform constants
{
	mat4 viewproj;
	//MaterialBuffer materialBuffer;
	uint padding[1 * 2];
	ObjectBuffer objectBuffer;
	VertexBuffer vertexBuffer;
} pc;

layout (location = 0) out vec2 outUV;
layout (location = 1) flat out uint outMaterialID;

void main()
{
	ObjectData o = pc.objectBuffer.objects[gl_InstanceIndex]; // gl_InstanceIndex from drawIndirectCommand
	Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
	
	vec3 pos = vec3(v.px, v.py, v.pz);
	vec4 worldPos = vec4(rotateQuat(pos, o.orientation) * o.scale + o.translation, 1.0);
	
	gl_Position = pc.viewproj * worldPos;
	
	outUV = vec2(v.uv_x, v.uv_y);
	outMaterialID = o.materialID;
}