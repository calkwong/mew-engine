#pragma once

#include "vk_math.h"

#include <initializer_list>
#include <vector>
#include <array>

struct IBLPushConstants
{
	glm::vec2 image_size{};
	uint32_t texture_id{};
	uint32_t image_id{};
	float roughness{};
};

struct LuminanceBinsPC
{
	VkDeviceAddress luminance_buffer{};
	VkDeviceAddress luminance_avg_buffer{};
	glm::vec2 screen_size{};
	uint32_t image_id{};
	float min_log_luminance{};
	float one_over_log_luminance_range{};
	uint32_t pixel_count{};
	float tau{};
	float delta_time{};
};

struct TonemapPC
{
	VkDeviceAddress luminance_avg_buffer{};
	glm::vec2 screen_size{};
	uint32_t image_id{};
	uint32_t autoexpose{};
	uint32_t tonemap_func{};
};

struct SHPushConstants
{
	VkDeviceAddress sh_buffer_address{};
	uint32_t cubemap_id{};
};

struct GPUPushConstants // temporarily shared by vertex and mesh shading path
{
	VkDeviceAddress object_buffer_address{};
	VkDeviceAddress vertex_buffer_address{};
	VkDeviceAddress meshtask_buffer_address{};
	VkDeviceAddress meshlet_buffer_address{};
	VkDeviceAddress meshlet_indices_buffer_address{};
	VkDeviceAddress cluster_indices_address{};
	VkDeviceAddress material_buffer_address{};
	VkDeviceAddress oit_buffer_address{};
	uint32_t debug_meshlets{};
	uint32_t padding{};
	glm::vec2 jitter_offset{}; // last + current frame jitter; should move this up but i am too lazy to edit shaders
};

struct DeferredPushConstants
{
	glm::vec4 cluster_size{}; // xyz are cluster data structure dimensions, w is a single cluster's dimension
	glm::vec2 screen_size{};
	VkDeviceAddress light_buffer_address{};
	VkDeviceAddress light_index_buffer_address{};
	VkDeviceAddress light_grid_buffer_address{};
	VkDeviceAddress oit_buffer_address{};
	VkDeviceAddress meshlet_indices_address{}; // TESTING FOR VIS BUFFER ONLY
	VkDeviceAddress meshlet_buffer_address{}; // TESTING FOR VIS BUFFER ONLY
	VkDeviceAddress vertex_buffer_address{}; // TESTING FOR VIS BUFFER ONLY
	VkDeviceAddress object_buffer_address{}; // TESTING FOR VIS BUFFER ONLY
	VkDeviceAddress material_buffer_address{}; // TESTING FOR VIS BUFFER ONLY
	VkDeviceAddress sh_buffer_address{};
	uint32_t depth_id{};
	uint32_t gbuffer_id{};
	uint32_t shadow_id{};
	uint32_t light_culling{}; // for toggling light culling between naive and proper implementation
	float near{};
	float scale{};
	float bias{};
	uint32_t debug_meshlets{};
	uint32_t resolve_transparent{};
	uint32_t shadows{};
	uint32_t pcf{};
	uint32_t debug_shadowmap{};
	uint32_t debug_cascades{};
	float max_prefiltered_lod{};
	float metallic{};
	float roughness{};
	uint32_t debug{};
	uint32_t map{};
};

// rasteroze shadows
struct ShadowPushConstants
{
	glm::mat4 viewproj{};
	VkDeviceAddress material_buffer_address{};
	VkDeviceAddress object_buffer_address{};
	VkDeviceAddress vertex_buffer_address{};
};

struct ShadowCullPushConstants
{
	VkDeviceAddress object_buffer_address{};
	VkDeviceAddress mesh_buffer_address{};
	VkDeviceAddress indices_buffer_address{};
	VkDeviceAddress draw_buffer_address{};
	uint32_t count{};
	uint32_t lod_enabled{};
};

struct SkyboxPushConstants
{
	glm::mat4 inverse_viewproj{};
	uint32_t texture_id{};
};

struct TAAResolvePC
{
	glm::vec2 screen_size{};
	glm::vec2 current_jitter{};
	uint32_t color_id{};
	uint32_t accum_id{}; // previous frame's accumulation/history buffer
	uint32_t depth_id{};
	uint32_t velocity_id{};
	uint32_t variance_clipping{};
	uint32_t history_filter{};
	uint32_t local_filter{};
	uint32_t ycocg{};
	uint32_t depth_dilation{};
	uint32_t weigh_luminance{};
};

struct DepthPyramidPushConstants
{
	std::array<int32_t, 2> image_size{};
	uint32_t texture_id{};
	uint32_t image_id{};
	uint32_t lod{};
};

struct ClusterGridPushConstants
{
	glm::mat4 inverse_proj{};
	VkDeviceAddress light_cluster_buffer_address{};
	glm::vec2 screen_size{};
	float cluster_dim{};
	float near{};
	float far{};
};

struct LightCullingPushConstants
{
	glm::mat4 view{};
	glm::mat4 light_rot{};
	VkDeviceAddress light_cluster_buffer_address{};
	VkDeviceAddress light_buffer_address{};
	VkDeviceAddress light_index_buffer_address{};
	VkDeviceAddress light_grid_buffer_address{};
	VkDeviceAddress light_count_buffer_address{};
};

struct PostFXPushConstants
{
	uint32_t texture_id{};
};

// TODO: rename?
struct ShaderPass
{
	VkPipeline pipeline{};
	VkPipelineLayout layout{};
};

struct ShaderProgram
{
	VkShaderModule module{};
	VkShaderStageFlagBits stage{};
};

struct PipelineBuilder
{
	std::vector<VkPipelineShaderStageCreateInfo> shader_stages{};
	std::vector<VkDynamicState> dynamic_state{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

	VkPipelineInputAssemblyStateCreateInfo input_assembly{};
	VkPipelineRasterizationStateCreateInfo rasterization{};
	VkPipelineColorBlendStateCreateInfo color_blend_info{};
	std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachment{};
	VkPipelineMultisampleStateCreateInfo multisampling{};
	VkPipelineLayout pipeline_layout{};
	VkPipelineDepthStencilStateCreateInfo depth_stencil{};
	VkPipelineRenderingCreateInfo render_info{};

	PipelineBuilder() { clear(); }

	void clear();

	VkPipeline build_pipeline(VkDevice device) const;
	void set_shaders(std::initializer_list<ShaderProgram*> programs);
	void set_input_topology(VkPrimitiveTopology topology);
	void set_polygon_mode(VkPolygonMode mode);
	void set_cull_mode(VkCullModeFlags cull_mode, VkFrontFace front_face);
	void set_multisampling_none();
	void set_blending_state(const std::vector<VkPipelineColorBlendAttachmentState>& blends);
	void set_color_attachment_format(const std::vector<VkFormat>& formats);
	void set_depth_format(VkFormat format);
	void disable_depth();
	void enable_depth(bool depth_write_enable, VkCompareOp op);

	// TODO: refactor into free functions
	VkPipelineColorBlendAttachmentState disable_blending();
	VkPipelineColorBlendAttachmentState enable_blending_additive();
	VkPipelineColorBlendAttachmentState enable_blending_alphablend();
};

struct ComputePipelineBuilder
{
	std::array<VkPipelineShaderStageCreateInfo, 1> shader_stages{};
	VkPipelineLayout pipeline_layout{};

	VkPipeline build_pipeline(VkDevice device) const;
	void set_shaders(const ShaderProgram* program);
};

namespace vkutil
{
	bool load_shader_module(const char* path, VkDevice device, VkShaderModule* out_shader_module);

	std::unique_ptr<ShaderPass> build_shader(VkDevice device, PipelineBuilder& builder, std::initializer_list<ShaderProgram*> programs, const std::vector<VkDescriptorSetLayout>& layouts, uint32_t pc_size);
	std::unique_ptr<ShaderPass> build_shader(VkDevice device, ComputePipelineBuilder& builder, const ShaderProgram* program, const std::vector<VkDescriptorSetLayout>& layouts, uint32_t pc_size);
} // namespace vkutil