#pragma once

#include "common.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vk_mem_alloc.h>

struct AllocatedImage
{
    VkImage image{};
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation{};
    VkExtent3D extent{};
    VkFormat format{};
};

struct AllocatedBuffer
{
    VkBuffer buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    VkDeviceSize size{};
};

AllocatedBuffer create_buffer(
    VmaAllocator allocator,
    size_t alloc_size,
    VmaAllocationCreateFlags flags,
    VkBufferUsageFlags usage,
    VkDeviceSize alignment = 0
);

AllocatedBuffer create_buffer_with_data(VkDevice device, VkQueue queue, VkFence fence, VkCommandPool command_pool, VkCommandBuffer cmd, VmaAllocator allocator, const void* data, size_t data_size, VkBufferUsageFlags = 0);

void destroy_buffer(VmaAllocator allocator, const AllocatedBuffer& buffer);

AllocatedImage create_image(VkDevice device, VmaAllocator allocator, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0, bool mipmapped = false);
AllocatedImage create_image_with_view(VkDevice device, VmaAllocator allocator, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0, bool mipmapped = false);
AllocatedImage create_3d_image(VkDevice device, VmaAllocator allocator, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0);
AllocatedImage upload_image(VkDevice device, VkQueue queue, VkFence fence, VkCommandPool command_pool, VkCommandBuffer cmd, VmaAllocator allocator, const void* data, uint32_t n_channels, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0, bool mipmapped = false);
AllocatedImage create_cubemap(VkDevice device, VmaAllocator allocator, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0, bool mipmapped = false);
void destroy_image(VkDevice device, VmaAllocator allocator, const AllocatedImage& image);

VkDeviceAddress get_buffer_address(VkDevice device, VkBuffer buffer);

namespace vkutil
{
void copy_image(VkCommandBuffer cmd, VkImage src, VkImage dst, VkExtent2D src_extent, VkExtent2D dst_extent);
void generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D extent, uint32_t layers = 1);
} // namespace vkutil

VkMemoryBarrier2 buffer_barrier(VkPipelineStageFlags2 src_stage_mask, VkPipelineStageFlags2 dst_stage_mask);
VkImageMemoryBarrier2 image_barrier(VkImage image, VkImageLayout old_layout, VkImageLayout new_layout, VkPipelineStageFlags2 src_stage_mask, VkPipelineStageFlags2 dst_stage_mask, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
void pipeline_barrier(VkCommandBuffer cmd, VkMemoryBarrier2* p_buffer, size_t count_buffer, VkImageMemoryBarrier2* p_image, size_t count_image);
void stage_barrier(VkCommandBuffer cmd, VkPipelineStageFlags2 src_stage_mask, VkPipelineStageFlags2 dst_stage_mask, VkAccessFlags2 src_access_mask, VkAccessFlags2 dst_access_mask);
void stage_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout, VkPipelineStageFlags2 src_stage_mask, VkPipelineStageFlags2 dst_stage_mask, VkAccessFlags2 src_access_mask, VkAccessFlags2 dst_access_mask, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
void stage_barrier(VkCommandBuffer cmd, VkPipelineStageFlags2 src_stage_mask, VkPipelineStageFlags2 dst_stage_mask);
void giga_barrier(VkCommandBuffer cmd);

void immediate_submit(VkDevice device, VkQueue queue, VkFence fence, VkCommandPool command_pool, VkCommandBuffer cmd, std::function<void(VkCommandBuffer cmd)>&& func);
