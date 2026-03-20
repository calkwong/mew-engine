#pragma once

#include "vk_scene.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// upadte
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

	// TODO: refactor in the future? added for multiple scenes compatibility
	// VkDeviceAddress material_buffer_address{};
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

// handles each LoadedGLTF
struct Loader
{
	std::vector<uint32_t> combined_indices{};
	std::vector<Vertex> combined_vertices{};

	std::vector<uint32_t> meshlet_indices{};
	std::vector<Meshlet> meshlets{};

	std::vector<MaterialData> materials{};
};

struct LoadedGLTF
{
	std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes{};
	std::unordered_map<std::string, std::shared_ptr<Node>> nodes{};
	std::unordered_map<std::string, AllocatedImage> images{};

	std::vector<std::shared_ptr<Node>> top_nodes{};

	VulkanEngine* creator{};
	std::string asset_path{};

	~LoadedGLTF() { clear(); }

private:
	void clear();
};

std::optional<std::shared_ptr<LoadedGLTF>> load_gltf(VulkanEngine* engine, Loader& loader, std::string& file_path);
