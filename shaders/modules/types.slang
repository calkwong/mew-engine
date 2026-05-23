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

public struct Vertex
{
	public float16_t px;
	public float16_t py;
	public float16_t pz;

	public uint16_t tangent;

	public uint normal;

	public float16_t uv_x;
	public float16_t uv_y;
};

public struct MaterialData
{
	public float4 base_color_factor;

	public float metallic_factor;
	public float roughness_factor;
	public uint diffuse_id;
	public uint metalroughness_id;

	public uint normal_id;
	public float3 emissive_factor;

	public uint emissive_id;
	public uint occlusion_id;
	public float transmission_factor;
	public uint transmission_id;
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

	public uint lod_count;
	public uint vertex_offset;

	public MeshLod lods[8];
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
};
