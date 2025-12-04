#pragma once

#include <vk_types.h>
#include <vk_pipelines.h>

#include "mikktspace.h"
#include <vulkan/vulkan.h>

#include <unordered_map>
#include <optional>
#include <vector>
#include <memory>
#include <string>

struct GeoSurface // rename this
{
	std::array<MeshLod, 8> mesh_lods{};
	uint32_t lod_count{};
	uint32_t vertex_offset{};

	uint32_t material{}; // master material handle
	uint32_t material_id{}; // for bindless material buffer
	
	MaterialPass pass{};
	Bounds bounds{}; // 28 bytes
	uint32_t meshlet_bits{};
};

struct MaterialInfo
{
	MaterialPass pass_type{};
	uint32_t double_sided{};
};

struct MeshAsset
{
	std::string name{};
	std::vector<GeoSurface> surfaces{};
	VkBuffer index_buffer{};
	VkDeviceAddress vertex_buffer_address{};

	// (!) refactor in the future? added for multiple scenes compatibility
	VkDeviceAddress material_buffer_address{};
};

struct Node
{
	std::shared_ptr<MeshAsset> mesh{};
	std::weak_ptr<Node> parent{};
	std::vector<std::shared_ptr<Node>> children{};

	glm::mat4 local_transform{};
	glm::mat4 world_transform{};

	void refresh_transform(const glm::mat4& parent_matrix)
	{
		world_transform = parent_matrix * local_transform;
		for (auto& c : children)
			c->refresh_transform(world_transform);
	}
};

class VulkanEngine;

struct LoadedGLTF
{
	std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes{};
	std::unordered_map<std::string, std::shared_ptr<Node>> nodes{};
	std::unordered_map<std::string, AllocatedImage> images{};

	std::vector<std::shared_ptr<Node>> top_nodes{};
	std::vector<VkSampler> samplers{};

	GPUMeshBuffers combined_mesh_buffer{};
	AllocatedBuffer material_buffer{}; // (!) possible refactor
	VkDeviceAddress material_buffer_address{};

	AllocatedBuffer meshlet_indices{};
	AllocatedBuffer meshlets{};

	VulkanEngine* creator{};

	~LoadedGLTF() { clear(); };

private:
	void clear();
};

std::optional<std::shared_ptr<LoadedGLTF>> load_gltf(VulkanEngine* engine, std::string_view file_path);

struct MikkMesh
{
	std::vector<Vertex>* vertices{};
	std::vector<uint32_t>* indices{};
};

void calculateTangents(MikkMesh& m);

int mikk_getNumFaces(const SMikkTSpaceContext* context);
int mikk_getNumVerticesOfFace(const SMikkTSpaceContext* context, int faceIndex);
void mikk_getPosition(const SMikkTSpaceContext* context, float outPosition[3], int faceIndex, int vertIndex);
void mikk_getNormal(const SMikkTSpaceContext* context, float outNormal[3], int faceIndex, int vertIndex);
void mikk_getTexCoord(const SMikkTSpaceContext* context, float outUV[2], int faceIndex, int vertIndex);
void mikk_setTSpaceBasic(const SMikkTSpaceContext* context, const float outTangent[3], float sign, int faceIndex, int vertIndex);