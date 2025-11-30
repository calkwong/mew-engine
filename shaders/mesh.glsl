struct ObjectData
{
	mat4 worldMatrix;
	uint materialID;
	uint padding[3];
};

struct MeshLod
{
	uint firstIndex;
	uint count;
	float error;
};

struct MeshData
{
	vec3 center;
	float radius;
	
	MeshLod lods[8];
	
	uint lodCount;
	uint padding[3];
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
	int vertexOffset;
	uint firstInstance;
};

struct Vertex {
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