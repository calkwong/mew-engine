struct ObjectData
{
	mat4 worldMatrix;
	uint materialID;
	uint meshletBitOffset;
	uint postPass;
	uint padding;
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

struct GPUInstance
{
	uint meshId;
	uint objectId;
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
	vec3 normal;
	float uv_y;
	vec4 tangent;
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