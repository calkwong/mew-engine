#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

struct ClusterAABB
{
	vec4 min;
	vec4 max;
};

struct PointLight
{
	vec4 pos;
	vec4 color;
};

struct LightGrid
{
	uint offset;
	uint count;
};

struct SH9
{
	float c[9];
};

struct ObjectData
{
	vec3 translation;
	float scale;
	vec4 orientation;

	uint mesh_id;
	uint material_id;
	uint meshlet_bit_offset;
	uint post_pass;
};

struct Meshlet
{
	float16_t cx, cy, cz;
	float16_t radius;

	uint data_offset; // index into MeshletIndices
	uint8_t vertex_count;
	uint8_t triangle_count;
	uint8_t padding[2];
};

struct MeshLod
{
	uint first_index;
	uint count;
	float error;
	uint meshlet_offset; // index into Meshlets, which has triangle count, vertices count and offset into Meshlet indices buffer
	uint meshlet_count;
};

struct MeshData
{
	vec3 center;
	float radius;

	MeshLod lods[8];

	uint lod_count;
	uint vertex_offset;
	uint padding[2];
};

struct DrawCommand
{
	uint index_count;
	uint instance_count;
	uint first_index;
	uint vertex_offset;
	uint first_instance;
};

struct MeshTaskCommand
{
	uint meshlet_offset;
	uint object_id;
	uint meshlet_visibility_offset;
	uint meshlet_count;
};

struct Vertex
{
	float16_t px;
	float16_t py;
	float16_t pz;

	uint16_t tangent;

	uint normal;

	float16_t uv_x;
	float16_t uv_y;
};

struct MaterialData
{
	vec4 base_color_factor;
	float metallic_factor;
	float roughness_factor;
	uint diffuse_id;
	uint metalroughness_id;
	uint normal_id;
	uint occlusion_id;
	uint emissive_id;
};

#ifdef SHADOW_CULL
#define MAX_DRAW_COMMANDS 400000
#else
#define MAX_DRAW_COMMANDS 1000000
// can raise this for testing millions of triangles, otherwise mesh path has artifacts (meshlets seem fine for ~1M meshes, likely self correction)
// for reasonable toy scenes, use 200000
#endif
struct DrawCommands
{
	uint opaque_count;
	uint alpha_clip_count;
	DrawCommand commands[MAX_DRAW_COMMANDS]; // shadow_cull is using 400000 as we cull then render opaque & alphaclip together. main view implements alphaclip as third pass (late only).
};

struct OITData
{
	uvec4 colors;
	uvec4 depths;
	vec4 transmissions; // could we pack this in color.a?
};
