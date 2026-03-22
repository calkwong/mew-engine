#pragma once

#include "vk_scene.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct MeshData
{
	std::array<MeshLod, 8> mesh_lods{};
	uint32_t lod_count{};
	uint32_t vertex_offset{};

	uint32_t material_id{};

	MaterialPass pass{};

	glm::vec3 center{};
	float radius{};

	uint32_t meshlet_bits{}; // equals # of meshlets for LOD 0, for tracking visibility
};

struct MeshAsset
{
	std::string name{};
	std::vector<MeshData> mesh{};
};

struct Node
{
	std::shared_ptr<MeshAsset> mesh_asset{};
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

	std::vector<uint32_t> indices{};
	std::vector<Vertex> vertices{};
	std::vector<uint32_t> meshlet_indices{};
	std::vector<Meshlet> meshlets{};
	std::vector<MaterialData> materials{};

	VulkanEngine* creator{};
	std::string asset_path{};

	~LoadedGLTF() { clear(); }

private:
	void clear();
};

std::optional<std::unique_ptr<LoadedGLTF>> load_gltf(VulkanEngine* engine, const std::string& file_path);
