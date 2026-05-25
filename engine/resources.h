#pragma once

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

AllocatedBuffer create_buffer(VmaAllocator allocator, size_t alloc_size, VmaAllocationCreateFlags flags, VkBufferUsageFlags usage);

AllocatedBuffer upload_buffer(
    VkDevice device,
    VkQueue queue,
    VkFence fence,
    VkCommandPool command_pool,
    VkCommandBuffer cmd,
    VmaAllocator allocator,
    const void* data,
    size_t data_size,
    VkBufferUsageFlags = 0
);

void destroy_buffer(VmaAllocator allocator, const AllocatedBuffer& buffer);

// this no longer creates a VkImageView
AllocatedImage create_image(
    VkDevice device,
    VmaAllocator allocator,
    VkExtent3D extent,
    VkFormat format,
    VkImageUsageFlags usage,
    VkImageAspectFlags aspect,
    VmaAllocationCreateFlags flags = 0,
    bool mipmapped = false
);

// temporary hack - this creates a VkImageView to submit as color attachment
AllocatedImage create_render_target(
    VkDevice device,
    VmaAllocator allocator,
    VkExtent3D extent,
    VkFormat format,
    VkImageUsageFlags usage,
    VkImageAspectFlags aspect,
    VmaAllocationCreateFlags flags = 0,
    bool mipmapped = false
);

AllocatedImage upload_image(
    VkDevice device,
    VkQueue queue,
    VkFence fence,
    VkCommandPool command_pool,
    VkCommandBuffer cmd,
    VmaAllocator allocator,
    const void* data,
    VkExtent3D extent,
    VkFormat format,
    VkImageUsageFlags usage,
    VkImageAspectFlags aspect,
    VmaAllocationCreateFlags flags = 0,
    bool mipmapped = false
);

AllocatedImage create_cubemap(
    VkDevice device,
    VmaAllocator allocator,
    VkExtent3D extent,
    VkFormat format,
    VkImageUsageFlags usage,
    VkImageAspectFlags aspect,
    VmaAllocationCreateFlags flags = 0,
    bool mipmapped = false
);

void destroy_image(VkDevice device, VmaAllocator allocator, const AllocatedImage& image);

VkDeviceAddress get_buffer_address(VkDevice device, VkBuffer buffer);

namespace vkutil
{
void copy_image(VkCommandBuffer cmd, VkImage src, VkImage dst, VkExtent2D src_extent, VkExtent2D dst_extent);

// assumes entire image begins in transfer_dst format, and returns in transfer_src format
void generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D extent, uint32_t layers = 1);
} // namespace vkutil

VkMemoryBarrier2 buffer_barrier(
    VkPipelineStageFlags2 src_stage_mask,
    VkPipelineStageFlags2 dst_stage_mask
);

VkImageMemoryBarrier2 image_barrier(
    VkImage image,
    VkImageLayout old_layout,
    VkImageLayout new_layout,
    VkPipelineStageFlags2 src_stage_mask,
    VkPipelineStageFlags2 dst_stage_mask,
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT
);

void pipeline_barrier(VkCommandBuffer cmd, VkMemoryBarrier2* p_buffer, size_t count_buffer, VkImageMemoryBarrier2* p_image, size_t count_image);

void stage_barrier(
    VkCommandBuffer cmd,
    VkPipelineStageFlags2 src_stage_mask,
    VkPipelineStageFlags2 dst_stage_mask,
    VkAccessFlags2 src_access_mask,
    VkAccessFlags2 dst_access_mask
);

void stage_barrier(
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

void stage_barrier(VkCommandBuffer cmd, VkPipelineStageFlags2 src_stage_mask, VkPipelineStageFlags2 dst_stage_mask);
void giga_barrier(VkCommandBuffer cmd);

VkSamplerCreateInfo get_sampler_info(
    VkFilter filter,
    VkSamplerAddressMode address,
    VkSamplerMipmapMode mipmap,
    VkSamplerReductionModeCreateInfo* reduce = 0
);

struct DescriptorImageInfo
{
    VkImage image{};
    VkFormat format{};
    VkImageViewType view_type{};
    VkImageAspectFlags aspect_flag{};
    uint32_t mip = 0;
};

void get_sample_descriptor(
    VkDevice device,
    VkFilter filter,
    VkSamplerMipmapMode mipmap,
    VkSamplerAddressMode address,
    VkSamplerReductionModeCreateInfo* reduce,
    void* descriptor,
    size_t descriptor_size
);

void get_image_descriptor(
    VkDevice device,
    VkImage image,
    VkFormat format,
    VkImageViewType view_type,
    VkImageAspectFlags aspect_flags,
    VkDescriptorType descriptor_type,
    void* descriptor,
    size_t descriptor_size,
    uint32_t mip = 0
);

void get_as_descriptor(
    VkDevice device,
    VkAccelerationStructureKHR as,
    VkDeviceSize as_size,
    void* descriptor,
    size_t descriptor_size
);

void get_buffer_descriptor(VkDevice device, AllocatedBuffer buffer, VkDescriptorType descriptor_type, void* descriptor, size_t descriptor_size);

void immediate_submit(VkDevice device, VkQueue queue, VkFence fence, VkCommandPool command_pool, VkCommandBuffer cmd, std::function<void(VkCommandBuffer cmd)>&& func);
