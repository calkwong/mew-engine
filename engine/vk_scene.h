#pragma once

#include "vk_types.h"
#include "vk_math.h"

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>
#include <array>

struct Material;
struct ShaderPass;
struct MeshLod;

struct alignas(16) DrawPrimitive {
	glm::vec3 center{};
	float radius{};

	std::array<MeshLod, 8> mesh_lods{};

	uint32_t lod_count{};
	uint32_t vertex_offset{};
	uint32_t padding[2];
};

template <typename T>
struct Handle
{
	uint32_t handle{};
};

template<>
struct Handle<DrawPrimitive> {
	uint32_t handle{};
};

struct RenderObject
{
	Handle<DrawPrimitive> primitive_id{};
	uint32_t material_id{};

	Material* material{};
	glm::mat4 transform{};

	VkDeviceAddress material_buffer_address{};
	uint32_t meshlet_bits{};
	uint32_t post_pass{};
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
	uint32_t mesh_id{};
	uint32_t object_id{};
};

struct ObjectData
{
	glm::mat4 transform{};
	uint32_t material_id{};
	uint32_t meshlet_bit_offset{};
	uint32_t post_pass{};
	uint32_t padding{}; 
};

struct CullData
{
	glm::mat4 view{};
	glm::vec4 frustum_planes{};
	VkDeviceAddress object_buffer_address{};
	VkDeviceAddress mesh_buffer_address{};
	VkDeviceAddress instance_buffer_address{};
	VkDeviceAddress draw_indirect_address{};
	VkDeviceAddress count_buffer_address{};
	VkDeviceAddress vis_buffer_address{};
	VkDeviceAddress meshtask_buffer_address{};
	uint32_t count{};
	uint32_t late{};
	uint32_t texture_id{};
	uint32_t occlusion_enabled{};

	float p00{};
	float p11{};
	float near{};
	float far{};

	glm::vec2 resolution{};
	float texture_lod{};
	float lod_distance_factor{};
	uint32_t lod_enabled{};
	uint32_t task_submit{};
	uint32_t post_pass{};
};

struct ClusterCullData
{
	glm::mat4 view{};
	glm::vec4 frustum_planes{};
	VkDeviceAddress object_buffer_address{};
	VkDeviceAddress meshlet_buffer_address{};
	VkDeviceAddress cluster_indices_address{};
	VkDeviceAddress cluster_count_address{};
	VkDeviceAddress count_buffer_address{};
	VkDeviceAddress cluster_vis_address{};
	VkDeviceAddress meshtask_buffer_address{};
	uint32_t count{};
	uint32_t late{};
	uint32_t texture_id{};
	uint32_t occlusion_enabled{};

	float p00{};
	float p11{};
	float near{};
	float far{};

	glm::vec2 resolution{};
	float texture_lod{};
	float lod_distance_factor{};
	uint32_t lod_enabled{};
	uint32_t task_submit{};
	uint32_t post_pass{};
};

struct PassObject
{
	ShaderPass* material{};
	Handle<DrawPrimitive> primitive_id{};
	Handle<RenderObject> renderable_id{}; // handle into renderables
	// TODO: implement hashing?
};

struct MeshTaskCommand
{
	uint32_t meshlet_offset{};
	uint32_t object_id{};
	uint32_t meshlet_visibility_offset{};
	uint32_t mesh_visibility{};
};

struct MeshAsset;

struct RenderScene
{
	enum class MeshPassType
	{
		Shadow,
		Forward,
		Transparent
	};

	struct MeshPass
	{
		std::vector<MultiBatch> multibatches{}; // unused during mesh shader test 
		std::vector<IndirectBatch> batches{}; // unused during mesh shader test 
		std::vector<uint32_t> unbatched_objects{}; // handles for renderables
		std::vector<PassObject> pass_objects{};

		AllocatedBuffer draw_indirect_buffer{};
		AllocatedBuffer meshtask_indirect_buffer{};
		AllocatedBuffer count_buffer{};
		AllocatedBuffer instance_buffer{};
		AllocatedBuffer vis_buffer{};
		AllocatedBuffer meshlet_vis_buffer{};

		AllocatedBuffer cluster_count_buffer{};
		AllocatedBuffer cluster_indices{};

		MeshPassType type{};
	};

	std::vector<RenderObject> renderables{};
	std::vector<DrawPrimitive> primitives{};
	std::unordered_map<MeshAsset*, Handle<DrawPrimitive>> mesh_cache{};

	GPUMeshBuffers combined_mesh_buffer{};
	AllocatedBuffer object_buffer{};
	AllocatedBuffer mesh_buffer{};
	AllocatedBuffer meshlet_buffer{};
	AllocatedBuffer meshlet_indices{};

	std::array<MeshPass, 4> shadow_pass{};
	MeshPass forward_pass{};
	MeshPass transparent_pass{};
	uint32_t total_meshlets_bits{};	  // should be per-pass
	uint32_t max_meshtask_commands{}; // should be per-pass

	void init();
	void build_mesh_buffer();
	void build_object_buffer();
	void build_pass_objects(MeshPass& pass);
	void sort_objects(MeshPass& pass); // sorts pass objects
	void build_indirect_batch(MeshPass& pass);
	void build_multi_batch(MeshPass& pass);
	void build_instance_buffer(MeshPass& pass);
};

