#pragma once

#include <initializer_list>
#include <filesystem>

struct ShaderPass
{
    VkPipeline pipeline{};
};

struct ShaderProgram
{
    VkShaderModule module{};
    std::string name{};
    std::filesystem::file_time_type time{};
};

using SpecConstants = std::initializer_list<uint32_t>;
using ShaderStages = std::initializer_list<VkShaderStageFlagBits>;
using ColorAttachmentFormats = std::initializer_list<VkFormat>;

bool load_shader_module(const char* path, VkDevice device, VkShaderModule* out_shader_module);

std::unique_ptr<ShaderPass> create_graphics_pipeline(
    VkDevice device,
    ShaderProgram* program,
    ShaderStages stages,
    VkShaderDescriptorSetAndBindingMappingInfoEXT* p_desc_set_and_binding_mapping_info,
    ColorAttachmentFormats color_attachment_formats,
    SpecConstants spec_constants = {},
    std::function<void(VkPipelineRasterizationStateCreateInfo& r, VkPipelineDepthStencilStateCreateInfo& d)>&& callback = {}
);

std::unique_ptr<ShaderPass> create_compute_pipeline(
    VkDevice device,
    ShaderProgram* program,
    VkShaderDescriptorSetAndBindingMappingInfoEXT* p_desc_set_and_binding_mapping_info,
    SpecConstants spec_constants = {}
);
