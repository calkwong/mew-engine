#pragma once

#include "vk_math.h"
#include "resources.h"

#include <array>
#include <vector>

struct Material; // TODO: not declared/anywhere
struct ShaderPass;

struct Vertex
{
	uint16_t px{};
	uint16_t py{};
	uint16_t pz{};

	uint16_t tangent{};

	uint32_t normal{};

	uint16_t uv_x{};
	uint16_t uv_y{};
};

enum class MaterialPass : uint32_t
{
	Opaque,
	Mask,
	Blend
};

struct MeshLod
{
	uint32_t first_index{};
	uint32_t count{};
	float error{};
	uint32_t meshlet_offset{};
	uint32_t meshlet_count{};
};

struct alignas(4) Meshlet
{
	uint16_t cx{};
	uint16_t cy{};
	uint16_t cz{};
	uint16_t radius{};

	uint32_t data_offset{}; // aka index first count

	uint8_t vertex_count{};
	uint8_t triangle_count{};
};

struct MaterialData
{
	glm::vec4 base_color_factor{ glm::vec4(1.0f) };
	float metallic_factor{ 1.0f };
	float roughness_factor{ 1.0f };
	uint32_t diffuse_id{};
	uint32_t metal_roughness_id{};
	uint32_t normal_id{};
	uint32_t occlusion_id{};
	uint32_t emissive_id{};
	uint32_t padding{};
};

struct SceneData
{
	glm::mat4 view{};
	glm::mat4 proj{};
	glm::mat4 viewproj{};
	glm::mat4 inverse_viewproj{};
	glm::mat4 previous_viewproj{};
	glm::mat4 light_rot{};
	std::array<glm::mat4, 4> shadow_transforms{};
	glm::vec4 cascade_splits{};
	glm::vec4 camera_pos{};
	glm::vec4 sunlight_color{};
	glm::vec4 sunlight_dir{};
	glm::vec4 textures{}; // cubemap/skybox, irradiance, prefiltered, brdf
	std::array<glm::mat4, 4> shadow_views{}; // for shadow_cull
	std::array<float, 4> shadow_widths{}; // TODO: move to pc?
};

struct CascadeData
{
	AllocatedImage shadow_map{};
	glm::mat4 viewproj{};
	float split_ratio{};
};

struct PointLight
{
	glm::vec4 pos{}; // pos & radius
	glm::vec4 color{};
};

struct ClusterAABB
{
	glm::vec4 min{};
	glm::vec4 max{};
};

struct LightGrid
{
	uint32_t offset{};
	uint32_t count{};
};

struct OITData
{
	glm::uvec4 colors{};
	glm::uvec4 depths{};
	glm::vec4 transmissions{};
};

struct alignas(16) Mesh
{
	glm::vec3 center{};
	float radius{};

	std::array<MeshLod, 8> mesh_lods{};

	uint32_t lod_count{};
	uint32_t vertex_offset{};
	uint32_t padding[2];
};

struct RenderObject
{
	glm::vec3 translation{};
	float scale{};
	glm::quat orientation{};

	uint32_t mesh_id{};
	uint32_t material_id{};
	uint32_t meshlet_bits{};
	uint32_t post_pass{};
};

struct ObjectData
{
	glm::vec3 translation{};
	float scale{};
	glm::quat orientation{};

	uint32_t mesh_id{};
	uint32_t material_id{};
	uint32_t meshlet_bit_offset{};
	uint32_t post_pass{};
};

struct CullData
{
	glm::mat4 view{};
	glm::vec4 frustum_planes{};
	VkDeviceAddress object_buffer_address{};
	VkDeviceAddress mesh_buffer_address{};
	VkDeviceAddress indices_buffer_address{};
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
		Opaque,
		Mask,
		Transparent
	};

	struct MeshPass
	{
		std::vector<uint32_t> unbatched_objects{}; // handles for renderables
		uint32_t indices_offset{};
		MeshPassType type{};
	};

	std::vector<RenderObject> renderables{};
	std::vector<Mesh> meshes{};
	std::unordered_map<MeshAsset*, uint32_t> mesh_cache{};

	AllocatedBuffer vertex_buffer{};
	AllocatedBuffer index_buffer{};
	AllocatedBuffer indices_buffer{}; // an indirection buffer - for indexing into the right RenderObject
	AllocatedBuffer object_buffer{};
	AllocatedBuffer mesh_buffer{};
	AllocatedBuffer meshlet_buffer{};
	AllocatedBuffer meshlet_indices{};
	AllocatedBuffer material_buffer{};

	AllocatedBuffer draw_indirect_buffer{};
	AllocatedBuffer meshtask_indirect_buffer{};
	AllocatedBuffer dispatch_buffer{};
	AllocatedBuffer vis_buffer{};
	AllocatedBuffer meshlet_vis_buffer{};

	AllocatedBuffer cluster_count_buffer{};
	AllocatedBuffer cluster_indices{};
	AllocatedBuffer oit_buffer{};

	AllocatedBuffer sh_buffer{};
	AllocatedBuffer luminance_buffer{};
	AllocatedBuffer luminance_avg_buffer{};

	MeshPass opaque_pass{};
	MeshPass mask_pass{};
	MeshPass transparent_pass{};
	uint32_t total_meshlets_bits{};
	uint32_t max_meshtask_commands{}; // should be per-pass? or not?

	void init();
};
