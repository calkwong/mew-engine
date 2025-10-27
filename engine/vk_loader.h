#pragma once

#include <vk_types.h>
#include <vk_pipelines.h>
#include <unordered_map>
#include <filesystem>
#include <vk_descriptors.h>

#include "mikktspace.h"

struct Bounds
{
	glm::vec3 origin{};
	float sphere_radius{};
	glm::vec3 extents{};
};

struct GeoSurface // rename this
{
	uint32_t start_index{};
	uint32_t count{};

	ShaderPass* material{};

	uint32_t material_id{};
	Bounds bounds{};

	MaterialPass pass{};
};

struct MaterialInfo
{
	uint8_t index{};
	MaterialPass pass_type{};
};

struct MeshAsset
{
	std::string name{};
	std::vector<GeoSurface> surfaces{};
	GPUMeshBuffers mesh_buffer{};

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

class DescriptorAllocatorGrowable;
class VulkanEngine;

struct LoadedGLTF
{
	std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes{};
	std::unordered_map<std::string, std::shared_ptr<Node>> nodes{};
	std::unordered_map<std::string, AllocatedImage> images{};
	//std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials{}; // obsolete?

	std::vector<std::shared_ptr<Node>> top_nodes{};
	std::vector<VkSampler> samplers{};

	// (!) obsolete in refactor? currently unused
	DescriptorAllocatorGrowable descriptor_pool{};

	AllocatedBuffer material_buffer{}; // refactor in new system?
	VkDeviceAddress material_buffer_address{};

	VulkanEngine* creator{};

	~LoadedGLTF() { clear(); };

private:
	void clear();
};


//std::optional<AllocatedImage> load_image(VulkanEngine* engine, fastgltf::Asset& asset, fastgltf::Image& image, bool mipmapped = false);
std::optional<std::shared_ptr<LoadedGLTF>> load_gltf(VulkanEngine* engine, std::string_view file_path, bool generate_tangents);

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