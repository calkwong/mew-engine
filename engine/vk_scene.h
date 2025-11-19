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
	Handle<DrawPrimitive> primitive_id{}; 
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
	uint32_t material_id{};
	glm::vec3 extent{};
	uint32_t padding{}; 
};

struct CullData
{
	std::array<glm::vec4, 6> frustum_planes{};
	VkDeviceAddress object_buffer_address{};
	VkDeviceAddress ginstance_buffer_address{};
	VkDeviceAddress indirect_buffer_address{};
	VkDeviceAddress instance_buffer_address{};
	uint32_t count{};
};

struct CompactIndirectData
{
	VkDeviceAddress indirect_buffer_address{};
	VkDeviceAddress compact_buffer_address{};
	VkDeviceAddress count_buffer_address{};
	uint32_t offsets[8]{}; // (!) TODO: refactor
	uint32_t count{};
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
	uint32_t pipeline_id{};
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

		AllocatedBuffer instance_buffer{};
		AllocatedBuffer ginstance_buffer{};
		AllocatedBuffer draw_indirect_buffer{};
		AllocatedBuffer clear_indirect_buffer{};
		AllocatedBuffer compact_indirect_buffer{};
		AllocatedBuffer count_buffer{};

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
	void build_ginstance_buffer(MeshPass& pass);
	void reset_indirect_buffer(MeshPass& pass, VkCommandBuffer cmd);
};

