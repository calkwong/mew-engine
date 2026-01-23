#pragma once

#include "vk_types.h"

#include <vulkan/vulkan.h>

struct ShaderPass
{
    VkPipeline pipeline{};
    VkPipelineLayout layout{};
};

struct Material
{
    ShaderPass* forward_pass{};
    ShaderPass* shadow_pass{};
};

struct PipelineBuilder
{
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages{};
    std::vector<VkDynamicState> dynamic_state{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    VkPipelineRasterizationStateCreateInfo rasterization{};
    VkPipelineColorBlendStateCreateInfo color_blend_info{};
    VkPipelineColorBlendAttachmentState color_blend_attachment{};
    std::vector<VkPipelineColorBlendAttachmentState> gbuffer_blend_attachment{};
    VkPipelineMultisampleStateCreateInfo multisampling{};
    VkPipelineLayout pipeline_layout{};
    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    VkPipelineRenderingCreateInfo render_info{};
    std::vector<VkFormat> color_attachment_format{};

    PipelineBuilder() { clear(); }

    void clear();

    VkPipeline build_pipeline(VkDevice device);
    void set_shaders(VkShaderModule vert_shader, VkShaderModule frag_shader);
    void set_shaders(VkShaderModule vert_shader);
    void set_mesh_shaders(VkShaderModule mesh_shader, VkShaderModule frag_shader);
    void set_mesh_shaders(VkShaderModule mesh_shader);
    void set_input_topology(VkPrimitiveTopology topology);
    void set_polygon_mode(VkPolygonMode mode);
    void set_cull_mode(VkCullModeFlags cull_mode, VkFrontFace front_face);
    void set_multisampling_none();
    void disable_blending();
    void set_blending_state(const VkPipelineColorBlendAttachmentState* states, size_t count);
    void set_color_attachment_format(VkFormat format);
    void set_gbuffer_format(VkFormat format, int count);
    void set_depth_format(VkFormat format);
    void disable_depth();
    void enable_depth(bool depth_write_enable, VkCompareOp op);
    void enable_blending_additive();
    void enable_blending_alphablend();
};

struct ComputePipelineBuilder
{
    std::array<VkPipelineShaderStageCreateInfo, 1> shader_stages{};
    VkPipelineLayout pipeline_layout{};

    VkPipeline build_pipeline(VkDevice device);
    void set_shaders(VkShaderModule comp_shader);
};

namespace vkutil
{
	bool load_shader_module(const char* path, VkDevice device, VkShaderModule* out_shader_module);

    std::unique_ptr<ShaderPass> build_shader(VkDevice device, PipelineBuilder& builder, std::vector<VkDescriptorSetLayout>& layouts, VkPushConstantRange* pc);

    std::unique_ptr<ShaderPass> build_shader(VkDevice device, ComputePipelineBuilder& builder, std::vector<VkDescriptorSetLayout>& layouts, VkPushConstantRange* pc);
}