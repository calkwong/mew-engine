module types;

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

public struct Dispatch
{
    public uint workgroup_x;
	public uint workgroup_y;
	public uint workgroup_z;
};

public struct Meshlet
{
	public float16_t cx, cy, cz;
	public float16_t radius;

	// public int8_t cone_axis[3];
	// public int8_t cone_cutoff;

	public uint data_offset; // index into MeshletIndices
	public uint8_t vertex_count;
	public uint8_t triangle_count;
	public uint8_t padding[2];
};
