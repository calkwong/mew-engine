#include "vk_pipelines.h"
#include "vk_initializers.h"
#include "vk_types.h"

#include <fmt/core.h>
#include <vulkan/vulkan.h>

#include <fstream>
#include <vector>

bool vkutil::load_shader_module(const char* path, VkDevice device, VkShaderModule* out_shader_module)
{
    // cursor at the end
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        return false;
    }

    // find what the size of the file is by looking up the location of the cursor
    // because the cursor is at the end, it gives the size directly in bytes
    size_t file_size = static_cast<size_t>(file.tellg());

    // spirv expects the buffer to be on uint32, so make sure to reserve a int
    // vector big enough for the entire file
    std::vector<uint32_t> buffer(file_size / sizeof(uint32_t)); 

    // put file cursor at beginning
    file.seekg(0);

    // load the entire file into the buffer
    file.read((char*)buffer.data(), file_size);

    // now that the file is loaded into the buffer, we can close it
    file.close();

    // create a new shader module, using the buffer we loaded
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

    // codeSize has to be in bytes, so multply the ints in the buffer by size of
    // int to know the real size of the buffer
    create_info.codeSize = buffer.size() * sizeof(uint32_t);
    create_info.pCode = buffer.data();

    // check that the creation goes well.
    VkShaderModule shader_module{};
    if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS) 
    {
        fmt::println("loading shader failed: {}", path);
        return false;
    }
    *out_shader_module = shader_module;
    return true;
}

void PipelineBuilder::clear()
{
    input_assembly = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };

    rasterization = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };

    color_blend_attachment = {};

    multisampling = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };

    pipeline_layout = {};

    depth_stencil = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };

    render_info = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };

    shader_stages.clear();
}

VkPipeline PipelineBuilder::build_pipeline(VkDevice device)
{
    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineColorBlendStateCreateInfo color_blend_info{};
    color_blend_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend_info.logicOpEnable = VK_FALSE;
    color_blend_info.logicOp = VK_LOGIC_OP_COPY;
    color_blend_info.attachmentCount = 1;
    color_blend_info.pAttachments = &color_blend_attachment;

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.pNext = &render_info;

    pipeline_info.stageCount = static_cast<uint32_t>(shader_stages.size());
    pipeline_info.pStages = shader_stages.data();
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterization;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pColorBlendState = &color_blend_info;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.layout = pipeline_layout;

    VkPipelineDynamicStateCreateInfo dynamic_info{};
    dynamic_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_info.pDynamicStates = dynamic_state.data();
    dynamic_info.dynamicStateCount = static_cast<uint32_t>(dynamic_state.size());

    pipeline_info.pDynamicState = &dynamic_info;

    VkPipeline pipeline{};
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS)
    {
        fmt::println("failed to create pipeline");
        return VK_NULL_HANDLE;
    }
    return pipeline;

}

void PipelineBuilder::set_shaders(VkShaderModule vert_shader, VkShaderModule frag_shader)
{
    shader_stages.clear();

    shader_stages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, vert_shader)
    );

    shader_stages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, frag_shader)
    );
}

void PipelineBuilder::set_shaders(VkShaderModule vert_shader)
{
    shader_stages.clear();

    shader_stages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, vert_shader)
    );
}

void PipelineBuilder::set_mesh_shaders(VkShaderModule mesh_shader, VkShaderModule frag_shader)
{
    shader_stages.clear();

    shader_stages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_MESH_BIT_EXT, mesh_shader)
    );

    shader_stages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, frag_shader)
    );
}

void PipelineBuilder::set_mesh_shaders(VkShaderModule mesh_shader)
{
    shader_stages.clear();

    shader_stages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_MESH_BIT_EXT, mesh_shader)
    );
}

void PipelineBuilder::set_input_topology(VkPrimitiveTopology topology)
{
    input_assembly.topology = topology;
    input_assembly.primitiveRestartEnable = VK_FALSE; // ?
}

void PipelineBuilder::set_polygon_mode(VkPolygonMode mode)
{
    rasterization.polygonMode = mode;
    rasterization.lineWidth = 1.0f; // ?
}

void PipelineBuilder::set_cull_mode(VkCullModeFlags cull_mode, VkFrontFace front_face)
{
    rasterization.cullMode = cull_mode;
    rasterization.frontFace = front_face;
}

void PipelineBuilder::set_multisampling_none()
{
    multisampling.sampleShadingEnable = VK_FALSE;
    // default to 1 sample per pixel
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f; // ?
    multisampling.pSampleMask = nullptr;
    // no alpha to coverage
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

}

void PipelineBuilder::disable_blending()
{
    color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.blendEnable = VK_FALSE;
}

void PipelineBuilder::set_color_attachment_format(VkFormat format)
{
    color_attachment_format = format;
    // connect format to render info
    render_info.colorAttachmentCount = (format == VK_FORMAT_UNDEFINED) ? 0 : 1;
    render_info.pColorAttachmentFormats = &color_attachment_format;
}

void PipelineBuilder::set_depth_format(VkFormat format)
{
    render_info.depthAttachmentFormat = format;
}

void PipelineBuilder::disable_depth()
{
    depth_stencil.depthTestEnable = VK_FALSE;
    depth_stencil.depthWriteEnable = VK_FALSE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_NEVER;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;
    depth_stencil.front = {};
    depth_stencil.back = {};
    depth_stencil.minDepthBounds = 0.0f; // ? verify if flipping not necessary for reverse z
    depth_stencil.maxDepthBounds = 1.0f;
}

void PipelineBuilder::enable_depth(bool depth_write_enable, VkCompareOp op)
{
    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = depth_write_enable;
    depth_stencil.depthCompareOp = op;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;
    depth_stencil.front = {};
    depth_stencil.back = {};
    depth_stencil.minDepthBounds = 0.0f; // ? verify if flipping not necessary for reverse z
    depth_stencil.maxDepthBounds = 1.0f;
}

void PipelineBuilder::enable_blending_additive()
{
    color_blend_attachment.blendEnable = VK_TRUE;
    color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
}

void PipelineBuilder::enable_blending_alphablend() // review alpha blend eq
{
    color_blend_attachment.blendEnable = VK_TRUE;
    color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    //color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
}

std::unique_ptr<ShaderPass> vkutil::build_shader(VkDevice device, PipelineBuilder& builder, std::vector<VkDescriptorSetLayout>& layouts, VkPushConstantRange* pc)
{
    std::unique_ptr<ShaderPass> shader = std::make_unique<ShaderPass>();

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(layouts.size());
    pipeline_layout_info.pSetLayouts = layouts.data();
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = pc;

    vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &shader->layout);

    builder.pipeline_layout = shader->layout;

    shader->pipeline = builder.build_pipeline(device);

    return shader;
}

VkPipeline ComputePipelineBuilder::build_pipeline(VkDevice device)
{
    VkPipeline pipeline{};
    
    VkComputePipelineCreateInfo compute_info{};
    compute_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_info.stage = shader_stages[0];
    compute_info.layout = pipeline_layout;

    vkCreateComputePipelines(device, 0, 1, &compute_info, nullptr, &pipeline);

    return pipeline;
}

void ComputePipelineBuilder::set_shaders(VkShaderModule comp_shader)
{
    shader_stages[0] = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, comp_shader);
}

std::unique_ptr<ShaderPass> vkutil::build_shader(VkDevice device, ComputePipelineBuilder& builder, std::vector<VkDescriptorSetLayout>& layouts, VkPushConstantRange* pc)
{
    std::unique_ptr<ShaderPass> shader = std::make_unique<ShaderPass>();

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(layouts.size());
    pipeline_layout_info.pSetLayouts = layouts.data();
    pipeline_layout_info.pushConstantRangeCount = pc != nullptr ? 1 : 0;
    pipeline_layout_info.pPushConstantRanges = pc;

    vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &shader->layout);

    builder.pipeline_layout = shader->layout;

    shader->pipeline = builder.build_pipeline(device);

    return shader;
}