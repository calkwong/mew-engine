#extension GL_EXT_buffer_reference : require

#include "types.glsl"

layout(buffer_reference, std430) buffer ClusterBuffer
{
	AABB aabb[];
};

layout(buffer_reference, std430) buffer LightBuffer
{
	PointLight lights[];
};

layout(buffer_reference, std430) buffer LightIndexBuffer
{
	uint indices[];
};

layout(buffer_reference, std430) buffer LightGridBuffer
{
	LightGrid grid[];
};

layout(buffer_reference, std430) buffer LightCountBuffer
{
	uint count;
};

layout(buffer_reference, std430) buffer SHBuffer
{
	SH9 rCoefficients;
	SH9 gCoefficients;
	SH9 bCoefficients;
};

layout(buffer_reference, std430) buffer LuminanceBuffer
{
	uint bins[256];
};

layout(buffer_reference, std430) buffer LuminanceAvgBuffer
{
	float luminanceAvg;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer
{ 
	Vertex vertices[];
};

layout(buffer_reference, std430) buffer ObjectBuffer
{ 
	ObjectData objects[];
};

layout(buffer_reference, std430) buffer MeshBuffer
{ 
	MeshData meshes[];
};

layout(buffer_reference, std430) buffer IndicesBuffer
{ 
	uint indices[];
};

layout(buffer_reference, std430) buffer DrawCommandsBuffer
{ 
	DrawCommands draws[4];
};

layout(buffer_reference, std430) buffer DispatchBuffer
{
	uint workgroupX;
	uint workgroupY;
	uint workgroupZ;
};

layout(buffer_reference, std430) buffer MeshTaskBuffer
{ 
	MeshTaskCommand commands[];
};
	
layout(buffer_reference, std430) buffer VisibilityBuffer
{
	uint visible[];
};

layout(buffer_reference, std430) buffer ClusterIndicesBuffer
{ 
	uint indices[];
};

layout(buffer_reference, std430) buffer ClusterCountBuffer
{
	uint workgroupX;
	uint workgroupY;
	uint workgroupZ;
};

layout(buffer_reference, std430) buffer MeshletBuffer
{ 
	Meshlet meshlets[];
};

#ifdef OIT_RESOLVE
#define OIT_QUALIFIER readonly
#else
#define OIT_QUALIFIER coherent
#endif
layout(buffer_reference, std430) OIT_QUALIFIER buffer OITBuffer
{ 
	OITData frags[];
};

layout(buffer_reference, std430) buffer MeshletIndicesBuffer
{ 
	uint indices[];
};

layout(buffer_reference, std430) buffer MaterialBuffer
{ 
	MaterialData materials[];
};