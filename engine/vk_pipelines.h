#pragma once

#include "vk_math.h"

#include <initializer_list>
#include <iterator>
#include <vector>
#include <array>
#include <filesystem>

// TODO: rename?
struct ShaderPass
{
    VkPipeline pipeline{};
    VkPipelineLayout layout{};
};

struct ShaderProgram
{
    VkShaderModule module{};
    std::string name{};
    std::filesystem::file_time_type time{};
};

using SpecConstants = std::initializer_list<uint32_t>;
using ShaderEntries = std::initializer_list<const char*>;
using ShaderStages = std::initializer_list<VkShaderStageFlagBits>;

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
    std::vector<VkDescriptorSetLayout> descriptor_layouts{};
    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    VkPipelineRenderingCreateInfo render_info{};
    std::string name{};

    PipelineBuilder()
    {
        clear();
    }

    void clear();

    VkPipeline build_pipeline(VkDevice device) const;
    void set_shaders(std::initializer_list<ShaderProgram*> programs, ShaderStages stages, ShaderEntries entries, VkShaderDescriptorSetAndBindingMappingInfoEXT* desc_set_and_binding_mapping_info);
    void set_input_topology(VkPrimitiveTopology topology);
    void set_polygon_mode(VkPolygonMode mode);
    void set_cull_mode(VkCullModeFlags cull_mode, VkFrontFace front_face);
    void set_multisampling_none();
    void set_blending_state(const std::vector<VkPipelineColorBlendAttachmentState>& blends);
    void set_color_attachment_format(const std::vector<VkFormat>& formats);
    void set_depth_format(VkFormat format);
    void disable_depth();
    void enable_depth(bool depth_write_enable, VkCompareOp op);
    void set_shader_specialization(VkSpecializationInfo* spec_info, size_t index = -1);
    void set_descriptor_layouts(std::initializer_list<VkDescriptorSetLayout> layouts);
    std::unique_ptr<ShaderPass> create_pipeline(
        VkDevice device,
        std::initializer_list<ShaderProgram*> program,
        ShaderStages stages,
        ShaderEntries entries = {},
        SpecConstants constants = {},
        VkShaderDescriptorSetAndBindingMappingInfoEXT* info = 0
    );

    // TODO: refactor into free functions
    VkPipelineColorBlendAttachmentState disable_blending();
    VkPipelineColorBlendAttachmentState enable_blending_additive();
    VkPipelineColorBlendAttachmentState enable_blending_alphablend();
};

struct ComputePipelineBuilder
{
    std::array<VkPipelineShaderStageCreateInfo, 1> shader_stages{};
    VkPipelineLayout pipeline_layout{};
    std::vector<VkDescriptorSetLayout> descriptor_layouts{};
    std::string name{};

    VkPipeline build_pipeline(VkDevice device) const;
    void set_shaders(const ShaderProgram* program, VkShaderDescriptorSetAndBindingMappingInfoEXT* desc_set_and_binding_mapping_info);
    void set_shader_specialization(VkSpecializationInfo* spec_info);
    void set_descriptor_layouts(std::initializer_list<VkDescriptorSetLayout> layouts);
    std::unique_ptr<ShaderPass> create_pipeline(VkDevice device, const ShaderProgram* program, SpecConstants constants = {}, VkShaderDescriptorSetAndBindingMappingInfoEXT* info = 0);
};

namespace vkutil
{
bool load_shader_module(const char* path, VkDevice device, VkShaderModule* out_shader_module);
} // namespace vkutil
