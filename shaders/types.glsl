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
	
	uint meshID;
	uint materialID;
	uint meshletBitOffset;
	uint postPass;
};

struct Meshlet
{
	float16_t cx, cy, cz;
	float16_t radius;
	
	uint dataOffset; // index into MeshletIndices
	uint8_t vertexCount; 
	uint8_t triangleCount; 
	uint8_t padding[2];
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
	DrawCommand commands[MAX_DRAW_COMMANDS]; // shadow_cull is using 400000 as we cull then render opaque & alphaclip together. main view implements alphaclip as third pass (late only).
};

struct OITData
{
	uvec4 colors;
	uvec4 depths;
	vec4 transmissions; // could we pack this in color.a?
};