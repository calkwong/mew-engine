#pragma once

#include <vk_types.h>

#include <vulkan/vulkan.h>
#include "glm/ext.hpp"

#include <memory>
#include <vector>
#include <array>

struct Material;
struct ShaderPass;

struct DrawPrimitive {
	uint32_t start_index{};
	uint32_t count{};
};

template<>
struct Handle<DrawPrimitive> {
	uint32_t handle{};
};

// (!) reorder
struct RenderObject
{
	Handle<DrawPrimitive> primitive_id{};
	uint32_t material_id{};

	Material* material{};
	glm::mat4 transform{};

	Bounds bounds{};

	VkDeviceAddress material_buffer_address{};
};

template<>
struct Handle<RenderObject> {
	uint32_t handle{};
};

struct IndirectBatch
{
	//Handle<DrawPrimitive> primitive_id{}; 
	ShaderPass* material{};
	uint32_t first{}; // refers to pass object array
	uint32_t count{}; // refers to pass object array
};

struct MultiBatch
{
	ShaderPass* pipeline{};
	uint32_t offset{}; // buffer offset for compact indirect buffer
	uint32_t max_draw_count{};
};

struct GPUInstance
{
	uint32_t object_id{};
	uint32_t batch_id{};
};

struct CompactInstance
{
	uint32_t object_id{};
};

struct ObjectData
{
	glm::mat4 transform{};
	glm::vec3 origin{};
	float radius{};
	uint32_t material_id{};
	//glm::vec3 extent{};
	uint32_t padding[3]{}; 
};

struct CullData
{
	std::array<glm::vec4, 6> frustum_planes{};
	glm::mat4 view{};
	VkDeviceAddress object_buffer_address{};
	VkDeviceAddress clear_indirect_address{};
	VkDeviceAddress draw_indirect_address{};
	VkDeviceAddress count_buffer_address{};
	VkDeviceAddress vis_buffer_address{};
	VkDeviceAddress debug_buffer_address{};
	uint32_t count{};
	uint32_t late{};
	uint32_t texture_id{};
	uint32_t occlusion{};

	glm::vec2 resolution{};
	float lod{};

	float p00{};
	float p11{};
	float near{};

};

struct PassObject
{
	ShaderPass* material{};
	Handle<DrawPrimitive> primitive_id{};
	Handle<RenderObject> renderable_id{}; // handle into renderables
	// (!) to do hash
};

struct GPUIndirect
{
	VkDrawIndexedIndirectCommand command{};
	//uint32_t object_id{};
};

struct MeshAsset;

struct RenderScene // (!) forward only for now
{
	enum class MeshPassType
	{
		Shadow,
		Forward,
		Transparent
	};

	struct MeshPass
	{
		std::vector<MultiBatch> multibatches{};
		std::vector<IndirectBatch> batches{};
		std::vector<uint32_t> unbatched_objects{}; // handles for renderables
		std::vector<PassObject> pass_objects{};

		AllocatedBuffer draw_indirect_buffer{};
		AllocatedBuffer clear_indirect_buffer{};
		AllocatedBuffer count_buffer{};
		AllocatedBuffer vis_buffer{};
		AllocatedBuffer debug_buffer{};

		MeshPassType type{};
	};

	std::vector<RenderObject> renderables{};
	std::vector<DrawPrimitive> primitives{};
	std::unordered_map<MeshAsset*, Handle<DrawPrimitive>> mesh_cache{};

	GPUMeshBuffers combined_mesh_buffer{};
	AllocatedBuffer object_buffer{}; 

	std::array<MeshPass, 4> shadow_pass{};
	MeshPass forward_pass{};
	MeshPass transparent_pass{};

	void init();
	void build_pass_objects(MeshPass& pass);
	void sort_objects(MeshPass& pass); // sorts pass objects
	void build_object_buffer();
	void build_indirect_batch(MeshPass& pass);
	void build_multi_batch(MeshPass& pass);
	void build_indirect_buffer(MeshPass& pass);
};

