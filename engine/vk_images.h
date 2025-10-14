#pragma once

#include <vulkan/vulkan.h>

namespace vkutil
{
	void transition_image(
		VkCommandBuffer cmd,
		VkImage image,
		VkImageLayout old_layout,
		VkImageLayout new_layout,
		VkPipelineStageFlags2 src_stage_mask,
		VkPipelineStageFlags2 dst_stage_mask,
		VkAccessFlags2 src_access_mask,
		VkAccessFlags2 dst_access_mask,
		VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT
	);

	// (!) via blitz
	void copy_image(VkCommandBuffer cmd, VkImage src, VkImage dst, VkExtent2D src_extent, VkExtent2D dst_extent);

	// may not work properly with array layers, not tested
	void generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D extent);
}