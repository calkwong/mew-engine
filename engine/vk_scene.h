#pragma once

#include <vk_types.h>

#include <vulkan/vulkan.h>
#include "glm/ext.hpp"

#include <memory>
#include <vector>

struct MeshAsset;
struct ShaderPass;
struct RenderObject;

struct DrawPrimitive
{
	uint32_t start_index{};
	uint32_t count{};
};

struct IndirectBatch
{
	//std::shared_ptr<MeshAsset> mesh{};
	uint32_t primitive{}; // handle
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

//struct ObjectData
//{
//	glm::mat4 transform{};
//	VkDeviceAddress vertex_buffer_address{};
//	uint32_t material_id{};
//	uint32_t padding{};
//};

struct CullData
{
	std::array<glm::vec4, 6> frustum_planes{};
	VkDeviceAddress object_buffer_address{};
	VkDeviceAddress ginstance_buffer_address{};
	uint32_t count{};
};

struct PassObject
{
	ShaderPass* material{};
	uint32_t primitive_id{};
	uint32_t renderables_id{}; // handle into renderables
	// (!) to do hash
};

struct RenderScene // (!) forward only for now
{
	std::vector<RenderObject> renderables{};
	std::vector<MultiBatch> multibatches{};
	std::vector<IndirectBatch> batches{};
	std::vector<size_t> unbatched_objects{}; // handles for renderables
	std::vector<PassObject> pass_objects{};
	std::vector<DrawPrimitive> primitives = { DrawPrimitive{0,0} };

	GPUMeshBuffers combined_mesh_buffer{};
	AllocatedBuffer object_buffer{}; 
	AllocatedBuffer instance_buffer{};
	AllocatedBuffer ginstance_buffer{};
	std::vector<VkDrawIndexedIndirectCommand> clear_indirect_buffer{};

	void build_pass_objects();
	void sort_objects(); // sorts pass objects
	void build_object_buffer();
	void build_indirect_batch();
	void build_multi_batch();
	void build_indirect_buffer();
	void build_ginstance_buffer();
	void build_instance_buffer();
	void reset_indirect_buffer(VkDrawIndexedIndirectCommand* draw_indirect_buffer);
	uint32_t add_primitive(uint32_t start_index, uint32_t count);
};

