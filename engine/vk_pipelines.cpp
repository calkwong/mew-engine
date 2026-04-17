#include "common.h"
#include "vk_pipelines.h"
#include "vk_initializers.h"
#include "vk_math.h"

#include <fmt/core.h>

#include <initializer_list>
#include <vector>
#include <array>
#include <fstream>
#include <cassert>

VkPipeline ComputePipelineBuilder::build_pipeline(VkDevice device) const
{
	VkPipeline pipeline{};

	VkComputePipelineCreateInfo compute_info{};
	compute_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	compute_info.stage = shader_stages[0];
	compute_info.layout = pipeline_layout;

	vkCreateComputePipelines(device, 0, 1, &compute_info, nullptr, &pipeline);

	return pipeline;
}

void ComputePipelineBuilder::set_shaders(const ShaderProgram* program)
{
	shader_stages[0] = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, program->module);
	name = "";
	name += program->name;
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

void PipelineBuilder::set_blending_state(const std::vector<VkPipelineColorBlendAttachmentState>& blends)
{
	color_blend_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blend_info.logicOpEnable = VK_FALSE;
	color_blend_info.logicOp = VK_LOGIC_OP_COPY;

	color_blend_info.attachmentCount = static_cast<uint32_t>(blends.size());
	color_blend_info.pAttachments = blends.data();
}

VkPipeline PipelineBuilder::build_pipeline(VkDevice device) const
{
	VkPipelineViewportStateCreateInfo viewport_state{};
	viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport_state.viewportCount = 1;
	viewport_state.scissorCount = 1;

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

void PipelineBuilder::set_shaders(std::initializer_list<ShaderProgram*> programs, ShaderStages stages, ShaderEntries entries)
{
    // TODO: uncomment after full slang port
    // assert(stages.size() == entries.size() && stages.size() > 0);

	shader_stages.clear();

	name = "";

	// TODO: after full slang port we will only have 1 shaderprogram
	auto program = programs.begin();
	for (auto it = stages.begin(); it != stages.end(); it++)
	{
		shader_stages.push_back(vkinit::pipeline_shader_stage_create_info(*it, (*program)->module));
		// TODO: better naming
		name += (*program)->name + '/';

		if (programs.size() == stages.size())
		    program++;
	}
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

void PipelineBuilder::set_color_attachment_format(const std::vector<VkFormat>& formats)
{
	// connect format to render info
	const uint32_t count = static_cast<uint32_t>(formats.size());
	const bool no_attachment = (count == 1) && (formats[0] == VK_FORMAT_UNDEFINED);
	render_info.colorAttachmentCount = no_attachment ? 0 : count;
	render_info.pColorAttachmentFormats = formats.data();
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

// TODO: refactor to free function
VkPipelineColorBlendAttachmentState PipelineBuilder::enable_blending_additive()
{
	VkPipelineColorBlendAttachmentState attachment_state{};

	attachment_state.blendEnable = VK_TRUE;
	attachment_state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	attachment_state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	attachment_state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
	attachment_state.colorBlendOp = VK_BLEND_OP_ADD;
	attachment_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	attachment_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	attachment_state.alphaBlendOp = VK_BLEND_OP_ADD;

	return attachment_state;
}

// TODO: refactor to free function
VkPipelineColorBlendAttachmentState PipelineBuilder::enable_blending_alphablend() // review alpha blend eq
{
	VkPipelineColorBlendAttachmentState attachment_state{};

	attachment_state.blendEnable = VK_TRUE;
	attachment_state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	attachment_state.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
	attachment_state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	attachment_state.colorBlendOp = VK_BLEND_OP_ADD;
	attachment_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	// color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	attachment_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	attachment_state.alphaBlendOp = VK_BLEND_OP_ADD;

	return attachment_state;
}

// TODO: refactor to free function
VkPipelineColorBlendAttachmentState PipelineBuilder::disable_blending()
{
	VkPipelineColorBlendAttachmentState attachment_state{};
	attachment_state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	attachment_state.blendEnable = VK_FALSE;

	return attachment_state;
}

void PipelineBuilder::set_descriptor_layouts(std::initializer_list<VkDescriptorSetLayout> layouts)
{
    for (auto layout : layouts)
        descriptor_layouts.push_back(layout);
}

void ComputePipelineBuilder::set_descriptor_layouts(std::initializer_list<VkDescriptorSetLayout> layouts)
{
    for (auto layout : layouts)
        descriptor_layouts.push_back(layout);
}

std::unique_ptr<ShaderPass> ComputePipelineBuilder::create_pipeline(VkDevice device, const ShaderProgram* program, SpecConstants constants)
{
	std::unique_ptr<ShaderPass> shader = std::make_unique<ShaderPass>();

	assert(program != nullptr);
	set_shaders(program);

	if (constants.size() != 0)
	{
		std::vector<VkSpecializationMapEntry> specialization_entries(constants.size());
		uint32_t index{};
		for (auto c : constants)
		{
			specialization_entries[index].constantID = index;
			specialization_entries[index].offset = index * static_cast<uint32_t>(sizeof(uint32_t));
			specialization_entries[index].size = static_cast<uint32_t>(sizeof(uint32_t));
			index++;
		}

		VkSpecializationInfo specialization_info{};
		specialization_info.mapEntryCount = static_cast<uint32_t>(specialization_entries.size());
		specialization_info.pMapEntries = specialization_entries.data();
		specialization_info.dataSize = constants.size() * sizeof(uint32_t);
		specialization_info.pData = constants.size() != 0 ? constants.begin() : nullptr;

		for (auto& shader_stage : shader_stages)
		{
			shader_stage.pSpecializationInfo = &specialization_info;
		}
	}

	VkPipelineLayoutCreateInfo pipeline_layout_info{};
	pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(descriptor_layouts.size());
	pipeline_layout_info.pSetLayouts = descriptor_layouts.data();

	auto push_constant_size = program->pc_size;

	VkPushConstantRange pc{};
	pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pc.size = push_constant_size;

	pipeline_layout_info.pushConstantRangeCount = push_constant_size != 0 ? 1 : 0;
	pipeline_layout_info.pPushConstantRanges = &pc;

	vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &shader->layout);

	pipeline_layout = shader->layout;

	shader->pipeline = build_pipeline(device);

	if (vkSetDebugUtilsObjectNameEXT)
	{
		VkDebugUtilsObjectNameInfoEXT name_info{};
		name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
		name_info.objectType = VK_OBJECT_TYPE_PIPELINE;
		name_info.objectHandle = (uint64_t)shader->pipeline;
		name_info.pObjectName = name.c_str();
		vkSetDebugUtilsObjectNameEXT(device, &name_info);
	}
	return shader;
}

std::unique_ptr<ShaderPass> PipelineBuilder::create_pipeline(VkDevice device, std::initializer_list<ShaderProgram*> program, ShaderStages stages, ShaderEntries entries, SpecConstants constants /*= {}*/)
{
    std::unique_ptr<ShaderPass> shader = std::make_unique<ShaderPass>();

	assert(program.size() > 0);
	set_shaders(program, stages, entries);
	std::vector<VkSpecializationMapEntry> specialization_entries(constants.size());

	if (constants.size() > 0)
	{
		uint32_t index{};
		for (auto c : constants)
		{
			specialization_entries[index].constantID = index;
			specialization_entries[index].offset = index * sizeof(uint32_t);
			specialization_entries[index].size = sizeof(uint32_t);
			index++;
		}
	}

	VkSpecializationInfo specialization_info{};
	specialization_info.mapEntryCount = static_cast<uint32_t>(specialization_entries.size());
	specialization_info.pMapEntries = specialization_entries.data();
	specialization_info.dataSize = constants.size() * sizeof(uint32_t);
	specialization_info.pData = constants.size() != 0 ? constants.begin() : nullptr;

	for (auto& shader_stage : shader_stages)
	{
		shader_stage.pSpecializationInfo = &specialization_info;
	}

	VkPipelineLayoutCreateInfo pipeline_layout_info{};
	pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(descriptor_layouts.size());
	pipeline_layout_info.pSetLayouts = descriptor_layouts.data();

	// TODO: clean up after full slang port
	uint32_t push_constant_size{};
	for (auto& p : program)
	{
        push_constant_size = p->pc_size;
	}

	VkPushConstantRange pc{};

	for (const auto& stage : stages)
	{
		pc.stageFlags |= stage;
	}
	pc.size = push_constant_size;

	pipeline_layout_info.pushConstantRangeCount = push_constant_size != 0 ? 1 : 0;
	pipeline_layout_info.pPushConstantRanges = &pc;

	vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &shader->layout);

	pipeline_layout = shader->layout;

	shader->pipeline = build_pipeline(device);

	// TODO: fix duplicated names for spec constants
	if (vkSetDebugUtilsObjectNameEXT)
	{
		VkDebugUtilsObjectNameInfoEXT name_info{};
		name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
		name_info.objectType = VK_OBJECT_TYPE_PIPELINE;
		name_info.objectHandle = (uint64_t)shader->pipeline;
		name_info.pObjectName = name.c_str();
		vkSetDebugUtilsObjectNameEXT(device, &name_info);
	}

	return shader;
}

bool vkutil::load_shader_module(const char* path, VkDevice device, VkShaderModule* out_shader_module)
{
	// cursor at the end
	std::ifstream file(path, std::ios::ate | std::ios::binary);

	if (!file.is_open())
	{
		return false;
	}

	// find what the size of the file is by looking up the location of the cursor
	// because the cursor is at the end, it gives the size directly in bytes
	const size_t file_size = file.tellg();

	// spirv expects the buffer to be on uint32, so make sure to reserve a int
	// vector big enough for the entire file
	std::vector<uint32_t> buffer(file_size / sizeof(uint32_t));

	// put file cursor at beginning
	file.seekg(0);

	// load the entire file into the buffer
	file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(file_size));

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

void PipelineBuilder::set_shader_specialization(VkSpecializationInfo* spec_info, size_t index /* = -1 */)
{
	if (index == -1)
	{
		for (auto& shader_stage : shader_stages)
		{
			shader_stage.pSpecializationInfo = spec_info;
		}
	}
	else
	{
		assert(index < shader_stages.size());
		shader_stages[index].pSpecializationInfo = spec_info;
	}
}

void ComputePipelineBuilder::set_shader_specialization(VkSpecializationInfo* spec_info)
{
	shader_stages[0].pSpecializationInfo = spec_info;
}
