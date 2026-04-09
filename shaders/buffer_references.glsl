#extension GL_EXT_buffer_reference : require

#include "types.glsl"

layout(buffer_reference, std430) buffer ClusterBuffer
{
	ClusterAABB aabb[];
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

// TODO: turn this into a single struct
layout(buffer_reference, std430) buffer SHBuffer
{
	SH9 r_coefficients;
	SH9 g_coefficients;
	SH9 b_coefficients;
};

layout(buffer_reference, std430) buffer LuminanceBuffer
{
	uint bins[256];
};

layout(buffer_reference, std430) buffer LuminanceAvgBuffer
{
	float luminance_avg;
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

layout(buffer_reference, std430) buffer MeshIndicesBuffer
{
	uint indices[];
};

layout(buffer_reference, std430) buffer DrawCommandsBuffer
{
	DrawCommands draws;
};

layout(buffer_reference, std430) buffer ShadowDrawCommandsBuffer
{
	DrawCommands draws[4];
};

layout(buffer_reference, std430) buffer DispatchBuffer
{
	uint workgroup_x;
	uint workgroup_y;
	uint workgroup_z;
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
	uint workgroup_x;
	uint workgroup_y;
	uint workgroup_z;
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

layout(buffer_reference, std430) buffer SpdCounterBuffer
{
    uint counter;
};
