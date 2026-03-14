#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// TODO: better naming, from cluster_grid
struct AABB
{
	vec4 min;
	vec4 max;
};

// TODO: better naming, from light_culling
struct PointLight // possible refactor to pos[], color[]?
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
	mat4 worldMatrix;
	uint meshID;
	uint materialID;
	uint meshletBitOffset;
	uint postPass;
};

struct Meshlet
{
	vec3 center;
	float radius;
	uint dataOffset; // index into MeshletIndices
	uint vertexCount;
	uint triangleCount;
	uint padding;
};

struct MeshLod
{
	uint firstIndex;
	uint count;
	float error;
	uint meshletOffset; // index into Meshlets, which has triangle count, vertices count and offset into Meshlet indices buffer
	uint meshletCount;
};

struct MeshData
{
	vec3 center;
	float radius;
	
	MeshLod lods[8];
	
	uint lodCount;
	uint vertexOffset;
	uint padding[2];
};

struct DrawCommand
{
	uint indexCount;
	uint instanceCount;
	uint firstIndex;
	uint vertexOffset;
	uint firstInstance;
};

struct MeshTaskCommand
{
	uint meshletOffset;
	uint objectId;
	uint meshletVisibilityOffset;
	uint meshletCount;
};

struct Vertex
{
	vec3 position;
	float uv_x;
	uint normal;
	float uv_y;
	
	uint16_t tangent;
	uint16_t padding[3];
	
	//uint padding[2];
	//vec4 tangent;
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

#ifdef SHADOW_CULL
#define MAX_DRAW_COMMANDS 400000
#else
#define MAX_DRAW_COMMANDS 200000
#endif
struct DrawCommands
{
	uint opaqueCount;
	uint alphaClipCount;
	DrawCommand commands[MAX_DRAW_COMMANDS]; // TODO: shadow_cull using 400000, why is it different again?
};

struct OITData
{
	uvec4 colors;
	uvec4 depths;
	vec4 transmissions; // could we pack this in color.a?
};