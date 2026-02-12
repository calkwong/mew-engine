#pragma once

#include "vk_types.h"

#include <initializer_list>
#include <vulkan/vulkan.h>

// TODO: rename?
struct ShaderPass
{
	VkPipeline pipeline{};
	VkPipelineLayout layout{};
};

// TODO: remove?
struct Material
{
	ShaderPass* forward_pass{};
	ShaderPass* shadow_pass{};
};

struct PipelineBuilder
{
	std::vector<VkPipelineShaderStageCreateInfo> shader_stages{};
	std::vector<VkDynamicState> dynamic_state{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

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
}