#pragma once

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>
#include "glm/ext.hpp"

struct MeshAsset;
struct ShaderPass;
struct RenderObject;

struct IndirectBatch
{
	std::shared_ptr<MeshAsset> mesh{};
	ShaderPass* forward_pass{};
	uint32_t first{};
	uint32_t count{};
};

struct MultiBatch
{
	uint32_t first{};
	uint32_t count{};
};

struct ObjectData
{
	glm::mat4 transform{};
	VkDeviceAddress vertex_buffer_address{};
	uint32_t material_id{};
	uint32_t padding{}; // (!)
};

std::vector<IndirectBatch> build_indirect_array(const std::vector<RenderObject>& renderables);

std::vector<MultiBatch> build_multibatch_array(std::vector<IndirectBatch>& batches);