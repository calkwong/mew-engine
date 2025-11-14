#pragma once

#include <vk_types.h>

#include <vulkan/vulkan.h>
#include "glm/ext.hpp"

#include <memory>
#include <vector>

struct Material;
struct ShaderPass;

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
	ShaderPass* forward_pass{};
	uint32_t first{}; // refers to pass object array
	uint32_t count{}; // refers to pass object array
};

struct MultiBatch
{
	uint32_t first{};
	uint32_t count{};
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

struct PassObject
{
	ShaderPass* material{};
	Handle<DrawPrimitive> primitive_id{};
	Handle<RenderObject> renderable_id{}; // handle into renderables
	// (!) to do hash
};

struct RenderScene // (!) forward only for now
{
	std::vector<RenderObject> renderables{};
	std::vector<MultiBatch> multibatches{};
	std::vector<IndirectBatch> batches{};
	std::vector<uint32_t> unbatched_objects{}; // handles for renderables
	std::vector<PassObject> pass_objects{};
	std::vector<DrawPrimitive> primitives{};

	GPUMeshBuffers combined_mesh_buffer{};
	AllocatedBuffer object_buffer{}; 
	AllocatedBuffer instance_buffer{};
	AllocatedBuffer ginstance_buffer{};
	AllocatedBuffer draw_indirect_buffer{};
	AllocatedBuffer clear_indirect_buffer{};

	void build_pass_objects();
	void sort_objects(); // sorts pass objects
	void build_object_buffer();
	void build_indirect_batch();
	void build_multi_batch();
	void build_indirect_buffer();
	void build_ginstance_buffer();
	void reset_indirect_buffer(VkCommandBuffer cmd);
};

