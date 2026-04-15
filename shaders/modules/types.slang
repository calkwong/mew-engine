module types;

/*
TODO:
- alternative for specifying public?
- replace preprocessor directives - enums?

*/

public struct ClusterAABB
{
	public float4 min;
	public float4 max;
};

public struct PointLight
{
	public float4 pos;
	public float4 color;
};

public struct LightGrid
{
	public uint offset;
	public uint count;
};

public struct ObjectData
{
	public float3 translation;
	public float scale;
	public float4 orientation;

	public uint mesh_id;
	public uint material_id;
	public uint meshlet_bit_offset;
	public uint post_pass;
};

public struct MeshLod
{
	public uint first_index;
	public uint count;
	public float error;

	// index into Meshlets, which has triangle count, vertices count and offset into Meshlet indices buffer
	public uint meshlet_offset;
	public uint meshlet_count;
};

public struct MeshData
{
	public float3 center;
	public float radius;

	public MeshLod lods[8];

	public uint lod_count;
	public uint vertex_offset;
	public uint padding[2];
};

public struct DrawCommand
{
	public uint index_count;
	public uint instance_count;
	public uint first_index;
	public uint vertex_offset;
	public uint first_instance;
};

// #ifdef SHADOW_CULL
// #define MAX_DRAW_COMMANDS 400000
// #else
// #define MAX_DRAW_COMMANDS 1000000
// #endif
public struct DrawCommands
{
	public uint opaque_count;
	public uint alpha_clip_count;

	// shadow_cull is using 400000 as we cull then render opaque & alphaclip together
	// main view implements alphaclip as third pass (late only)
	public DrawCommand commands[400000];
};
