#pragma once

#include <vk_types.h>

#include <vulkan/vulkan.h>

#include <span>

struct ShaderEffect
{
    std::array<VkShaderModule, 2> modules{};
    std::vector<VkDescriptorSetLayout> layouts{};
    std::vector<VkPushConstantRange> pc{};

    void build_effect(VkDevice device, const char* vert_path, const char* frag_path);
};

struct ShaderPass
{
    VkPipeline pipeline{};
    VkPipelineLayout layout{};
};

struct PipelineBuilder
{
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages{};

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    VkPipelineRasterizationStateCreateInfo rasterization{};
    VkPipelineColorBlendAttachmentState color_blend_attachment{};
    VkPipelineMultisampleStateCreateInfo multisampling{};
    VkPipelineLayout pipeline_layout{};
    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    VkPipelineRenderingCreateInfo render_info{};
    VkFormat color_attachment_format;

    PipelineBuilder() { clear(); }

    void clear();

    VkPipeline build_pipeline(VkDevice device);
    void set_shaders(VkShaderModule vert_shader, VkShaderModule frag_shader);
    void set_input_topology(VkPrimitiveTopology topology);
    void set_polygon_mode(VkPolygonMode mode);
    void set_cull_mode(VkCullModeFlags cull_mode, VkFrontFace front_face);
    void set_multisampling_none();
    void disable_blending();
    void set_color_attachment_format(VkFormat format);
    void set_depth_format(VkFormat format);
    void disable_depth();
    void enable_depth(bool depth_write_enable, VkCompareOp op);
    void enable_blending_additive();
    void enable_blending_alphablend();
};

namespace vkutil
{
	bool load_shader_module(const char* path, VkDevice device, VkShaderModule* out_shader_module);

    std::unique_ptr<ShaderPass> build_shader(VkDevice device, ShaderEffect* effect, PipelineBuilder& builder);
}