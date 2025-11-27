#version 450
#extension GL_EXT_buffer_reference : require

struct Vertex
{
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 tangent;
};

struct ObjectData
{
	mat4 worldMatrix;
	vec3 origin;
	uint materialID;
	vec3 extent;
	uint padding;
};

struct MaterialData
{
	vec4 baseColorFactor;
	float metallicFactor;
	float roughnessFactor;
	uint diffuseID;
	uint metalRoughnessID;
	uint normalID;
	uint occlusionID;
	uint emissiveID;
};

layout (buffer_reference, std430) readonly buffer VertexBuffer {
	Vertex vertices[];
};

layout(buffer_reference, std430) readonly buffer ObjectBuffer
{ 
	ObjectData objects[];
};

layout(buffer_reference, std430) readonly buffer InstanceBuffer
{ 
	uint instances[];
};

layout(buffer_reference, std430) readonly buffer MaterialBuffer
{ 
	MaterialData materials[];
};

layout (push_constant) uniform constants
{
	mat4 viewproj;
	MaterialBuffer materialBuffer;
	ObjectBuffer objectBuffer;
	VertexBuffer vertexBuffer;
} pc;

layout (location = 0) out vec2 outUV;
layout (location = 1) flat out uint outMaterialID;

void main()
{
	ObjectData o = pc.objectBuffer.objects[gl_InstanceIndex];
	Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
	
	vec4 position = o.worldMatrix * vec4(v.position, 1.0);
	gl_Position = pc.viewproj * position;
	
	outUV = vec2(v.uv_x, v.uv_y); // is this wasted?
	outMaterialID = o.materialID;
}