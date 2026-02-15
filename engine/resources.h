#pragma once

#include <vk_mem_alloc.h>

#include <functional>

struct AllocatedImage
{
	VkImage image{};
	VkImageView view{};
	VmaAllocation allocation{};
	VkExtent3D extent{};
	VkFormat format{};
};

struct AllocatedBuffer
{
	VkBuffer buffer{};
	VmaAllocation allocation{};
	VmaAllocationInfo info{};
};

// TODO: ideally in engine or device file, leaving here for convenience now as uploads rely on it
void immediate_submit(VkDevice device, VkQueue queue, VkCommandBuffer cmd, VkFence fence, std::function<void(VkCommandBuffer cmd)>&& func);

AllocatedBuffer create_buffer(VmaAllocator allocator, size_t alloc_size, VmaAllocationCreateFlags flags, VkBufferUsageFlags usage);
AllocatedBuffer upload_buffer(VkDevice device, VkQueue queue, VkCommandBuffer cmd, VkFence fence, VmaAllocator allocator, const void* data, size_t data_size, VkBufferUsageFlags = 0);
void destroy_buffer(VmaAllocator allocator, const AllocatedBuffer& buffer);

// view has access to all mip and layers
AllocatedImage create_image(VkDevice device, VmaAllocator allocator, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0, bool mipmapped = false);
AllocatedImage upload_image(VkDevice device, VkQueue queue, VkCommandBuffer cmd, VkFence fence, VmaAllocator allocator, const void* data, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0, bool mipmapped = false);
void destroy_image(VkDevice device, VmaAllocator allocator, const AllocatedImage& image);

VkDeviceAddress get_buffer_address(VkDevice device, VkBuffer buffer);

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

	void copy_image(VkCommandBuffer cmd, VkImage src, VkImage dst, VkExtent2D src_extent, VkExtent2D dst_extent);

	// assumes entire image begins in transfer_dst format, and returns in transfer_src format
	void generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D extent, uint32_t layers = 1);

	void transition_buffer(
	    VkCommandBuffer cmd,
	    VkPipelineStageFlags2 src_stage_mask,
	    VkPipelineStageFlags2 dst_stage_mask,
	    VkAccessFlags2 src_access_mask,
	    VkAccessFlags2 dst_access_mask
	);
}

