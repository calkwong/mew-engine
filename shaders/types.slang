#ifndef TYPES_SLANG
#define TYPES_SLANG

public struct ClusterAABB
{
	float4 min;
	float4 max;
};

struct PointLight
{
	float4 pos;
	float4 color;
};

struct LightGrid
{
	uint offset;
	uint count;
};

#endif
