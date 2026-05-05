#include "common.h"
#include "resources.h"
#include "vk_initializers.h"
#include "vk_engine.h"

#include <vk_mem_alloc.h>

void destroy_buffer(VmaAllocator allocator, const AllocatedBuffer& buffer)
{
	vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
}

AllocatedBuffer create_buffer(VmaAllocator allocator, size_t alloc_size, VmaAllocationCreateFlags flags, VkBufferUsageFlags usage)
{
	VkBufferCreateInfo buffer_info{};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size = alloc_size;
	buffer_info.usage = usage;

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
	alloc_info.flags = flags;

	AllocatedBuffer new_buffer{};

	VK_CHECK(vmaCreateBuffer(allocator, &buffer_info, &alloc_info, &new_buffer.buffer, &new_buffer.allocation, &new_buffer.info));

	return new_buffer;
}

AllocatedBuffer upload_buffer(VulkanEngine* engine, VmaAllocator allocator, const void* data, size_t data_size, VkBufferUsageFlags flags /*= 0*/)
{
	AllocatedBuffer buffer = create_buffer(allocator, data_size, 0, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | flags);

	AllocatedBuffer staging = create_buffer(allocator, data_size, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

	void* staging_data = staging.info.pMappedData;
	memcpy(staging_data, data, data_size);

	engine->immediate_submit([&](VkCommandBuffer cmd)
	{
		VkBufferCopy copy{};
		copy.dstOffset = 0;
		copy.srcOffset = 0;
		copy.size = data_size;

		vkCmdCopyBuffer(cmd, staging.buffer, buffer.buffer, 1, &copy);
	});

	destroy_buffer(allocator, staging);

	return buffer;
}

AllocatedImage create_image(VkDevice device, VmaAllocator allocator, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags /*= 0*/, bool mipmapped /*= false*/)
{
	AllocatedImage new_image{};
	new_image.extent = extent;
	new_image.format = format;

	VkImageCreateInfo img_info = vkinit::image_create_info(format, usage, extent);
	if (mipmapped)
	{
		img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;
		img_info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.flags = flags;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	VK_CHECK(vmaCreateImage(allocator, &img_info, &alloc_info, &new_image.image, &new_image.allocation, nullptr));

	VkImageViewCreateInfo img_view_info = vkinit::imageview_create_info(format, new_image.image, aspect);

	VK_CHECK(vkCreateImageView(device, &img_view_info, nullptr, &new_image.view));

	return new_image;
}

// currently used for HDR, png and jpg, NOT ktx2
AllocatedImage upload_image(VulkanEngine* engine, VkDevice device, VmaAllocator allocator, const void* data, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags, bool mipmapped)
{
	size_t data_size = extent.depth * extent.width * extent.height * 4; // 4 is # of channels
	if (format == VK_FORMAT_R32G32B32A32_SFLOAT) // TODO: hdr only?
		data_size *= sizeof(float);
	AllocatedBuffer upload_buffer = create_buffer(allocator, data_size, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

	memcpy(upload_buffer.info.pMappedData, data, data_size);

	// dst_bit to account for copy from staging buffer
	AllocatedImage new_image = create_image(device, allocator, extent, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT, aspect, flags, mipmapped);

	engine->immediate_submit([&](VkCommandBuffer cmd)
	{
		vkutil::transition_image(
		    cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		    0,
		    VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		    0,
		    VK_ACCESS_2_TRANSFER_WRITE_BIT
		);

		VkBufferImageCopy copy_region{};
		copy_region.bufferOffset = 0;
		copy_region.bufferRowLength = 0;
		copy_region.bufferImageHeight = 0;

		copy_region.imageSubresource.aspectMask = aspect;
		copy_region.imageSubresource.mipLevel = 0;
		copy_region.imageSubresource.baseArrayLayer = 0;
		copy_region.imageSubresource.layerCount = 1;

		copy_region.imageExtent = extent;

		vkCmdCopyBufferToImage(cmd, upload_buffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

		if (mipmapped)
		{
			vkutil::generate_mipmaps(cmd, new_image.image, VkExtent2D{ new_image.extent.width, new_image.extent.height });
		}
		else
		{
			vkutil::transition_image(
			    cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			    VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			    VK_ACCESS_2_TRANSFER_WRITE_BIT,
			    VK_ACCESS_2_SHADER_READ_BIT
			);
		}
	});

	destroy_buffer(allocator, upload_buffer);

	return new_image;
}

AllocatedImage create_cubemap(VkDevice device, VmaAllocator allocator, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags /*= 0*/, bool mipmapped /*= false*/)
{
	AllocatedImage new_image{};
	new_image.extent = extent;
	new_image.format = format;

	VkImageCreateInfo img_info{ vkinit::image_create_info(format, usage, new_image.extent) };
	img_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	img_info.arrayLayers = 6;

	if (mipmapped)
	{
		img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(extent.width, extent.height))))) + 1;
		img_info.usage |= (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	}

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.flags = flags;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	VK_CHECK(vmaCreateImage(allocator, &img_info, &alloc_info, &new_image.image, &new_image.allocation, nullptr));

	VkImageViewCreateInfo img_view_info{ vkinit::imageview_create_info(format, new_image.image, VK_IMAGE_ASPECT_COLOR_BIT) };
	img_view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;

	VK_CHECK(vkCreateImageView(device, &img_view_info, nullptr, &new_image.view));

	return new_image;
}

void destroy_image(VkDevice device, VmaAllocator allocator, const AllocatedImage& image)
{
	if (image.view == nullptr)
		fmt::println("was null");
	vkDestroyImageView(device, image.view, nullptr);
	vmaDestroyImage(allocator, image.image, image.allocation);
}

VkDeviceAddress get_buffer_address(VkDevice device, VkBuffer buffer)
{
	VkBufferDeviceAddressInfo address_info{};
	address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	address_info.buffer = buffer;
	return vkGetBufferDeviceAddress(device, &address_info);
}

void vkutil::transition_image(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout old_layout,
    VkImageLayout new_layout,
    VkPipelineStageFlags2 src_stage_mask,
    VkPipelineStageFlags2 dst_stage_mask,
    VkAccessFlags2 src_access_mask,
    VkAccessFlags2 dst_access_mask,
    VkImageAspectFlags aspect /*= VK_IMAGE_ASPECT_COLOR_BIT*/
)
{
	VkImageMemoryBarrier2 barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	barrier.srcStageMask = src_stage_mask;
	barrier.dstStageMask = dst_stage_mask;
	barrier.srcAccessMask = src_access_mask;
	barrier.dstAccessMask = dst_access_mask;
	barrier.oldLayout = old_layout;
	barrier.newLayout = new_layout;
	barrier.subresourceRange = vkinit::image_subresource_range(aspect);
	barrier.image = image;

	VkDependencyInfo info{};
	info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	info.imageMemoryBarrierCount = 1;
	info.pImageMemoryBarriers = &barrier;

	vkCmdPipelineBarrier2(cmd, &info);
}

void vkutil::copy_image(VkCommandBuffer cmd, VkImage src, VkImage dst, VkExtent2D src_extent, VkExtent2D dst_extent)
{
	VkImageBlit2 blit{};
	blit.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;

	blit.srcOffsets[1].x = src_extent.width;
	blit.srcOffsets[1].y = src_extent.height;
	blit.srcOffsets[1].z = 1;

	blit.dstOffsets[1].x = dst_extent.width;
	blit.dstOffsets[1].y = dst_extent.height;
	blit.dstOffsets[1].z = 1;

	blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	blit.srcSubresource.baseArrayLayer = 0;
	blit.srcSubresource.layerCount = 1;
	blit.srcSubresource.mipLevel = 0;

	blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	blit.dstSubresource.baseArrayLayer = 0;
	blit.dstSubresource.layerCount = 1;
	blit.dstSubresource.mipLevel = 0;

	VkBlitImageInfo2 blit_info{};
	blit_info.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
	blit_info.dstImage = dst;
	blit_info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	blit_info.srcImage = src;
	blit_info.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	blit_info.filter = VK_FILTER_LINEAR;
	blit_info.regionCount = 1;
	blit_info.pRegions = &blit;

	vkCmdBlitImage2(cmd, &blit_info);
}

void vkutil::generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D extent, uint32_t layers /*= 1*/)
{
	int mip_levels = static_cast<int>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;
	int current_layer = layers - 1;

	VkExtent2D image_size_copy = extent;

	while (current_layer >= 0)
	{
		for (int mip = 0; mip < mip_levels; mip++)
		{
			VkExtent2D half_size = image_size_copy;
			half_size.width /= 2;
			half_size.height /= 2;

			VkImageMemoryBarrier2 barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;

			barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
			barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
			barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;

			barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			barrier.image = image;

			VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.aspectMask = aspect;
			barrier.subresourceRange.baseArrayLayer = current_layer;
			barrier.subresourceRange.layerCount = 1;
			barrier.subresourceRange.baseMipLevel = mip;
			barrier.subresourceRange.levelCount = 1;

			VkDependencyInfo dep_info{};
			dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
			dep_info.imageMemoryBarrierCount = 1;
			dep_info.pImageMemoryBarriers = &barrier;

			vkCmdPipelineBarrier2(cmd, &dep_info);

			if (mip < mip_levels - 1)
			{
				VkImageBlit2 blit_region{};
				blit_region.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
				blit_region.srcOffsets[1].x = image_size_copy.width;
				blit_region.srcOffsets[1].y = image_size_copy.height;
				blit_region.srcOffsets[1].z = 1;

				blit_region.dstOffsets[1].x = half_size.width;
				blit_region.dstOffsets[1].y = half_size.height;
				blit_region.dstOffsets[1].z = 1;

				blit_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				blit_region.srcSubresource.baseArrayLayer = current_layer;
				blit_region.srcSubresource.layerCount = 1;
				blit_region.srcSubresource.mipLevel = mip;

				blit_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				blit_region.dstSubresource.baseArrayLayer = current_layer;
				blit_region.dstSubresource.layerCount = 1;
				blit_region.dstSubresource.mipLevel = mip + 1;

				VkBlitImageInfo2 blit_info{};
				blit_info.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
				blit_info.dstImage = image;
				blit_info.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
				blit_info.srcImage = image;
				blit_info.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
				blit_info.filter = VK_FILTER_LINEAR;
				blit_info.regionCount = 1;
				blit_info.pRegions = &blit_region;

				vkCmdBlitImage2(cmd, &blit_info);

				image_size_copy = half_size;
			}
		}
		current_layer--;
		image_size_copy = extent;
	}
}

void vkutil::transition_buffer(
    VkCommandBuffer cmd,
    VkPipelineStageFlags2 src_stage_mask,
    VkPipelineStageFlags2 dst_stage_mask,
    VkAccessFlags2 src_access_mask,
    VkAccessFlags2 dst_access_mask
)
{
	VkMemoryBarrier2 barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
	barrier.srcStageMask = src_stage_mask;
	barrier.dstStageMask = dst_stage_mask;
	barrier.srcAccessMask = src_access_mask;
	barrier.dstAccessMask = dst_access_mask;

	VkDependencyInfo info{};
	info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	info.memoryBarrierCount = 1;
	info.pMemoryBarriers = &barrier;
	vkCmdPipelineBarrier2(cmd, &info);
}

VkMemoryBarrier2 buffer_barrier(
    VkPipelineStageFlags2 src_stage_mask,
    VkPipelineStageFlags2 dst_stage_mask,
    VkAccessFlags2 src_access_mask,
    VkAccessFlags2 dst_access_mask
)
{
	VkMemoryBarrier2 barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
	barrier.srcStageMask = src_stage_mask;
	barrier.dstStageMask = dst_stage_mask;
	barrier.srcAccessMask = src_access_mask;
	barrier.dstAccessMask = dst_access_mask;
	return barrier;
}

VkImageMemoryBarrier2 image_barrier(
    VkImage image,
    VkImageLayout old_layout,
    VkImageLayout new_layout,
    VkPipelineStageFlags2 src_stage_mask,
    VkPipelineStageFlags2 dst_stage_mask,
    VkAccessFlags2 src_access_mask,
    VkAccessFlags2 dst_access_mask,
    VkImageAspectFlags aspect
)
{
	VkImageMemoryBarrier2 barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	barrier.srcStageMask = src_stage_mask;
	barrier.dstStageMask = dst_stage_mask;
	barrier.srcAccessMask = src_access_mask;
	barrier.dstAccessMask = dst_access_mask;
	barrier.oldLayout = old_layout;
	barrier.newLayout = new_layout;
	barrier.subresourceRange = vkinit::image_subresource_range(aspect);
	barrier.image = image;
	return barrier;
}

void pipeline_barrier(VkCommandBuffer cmd, VkMemoryBarrier2* p_buffer, size_t count_buffer, VkImageMemoryBarrier2* p_image, size_t count_image)
{
	VkDependencyInfo info{};
	info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	info.memoryBarrierCount = static_cast<uint32_t>(count_buffer);
	info.pMemoryBarriers = p_buffer;
	info.imageMemoryBarrierCount = static_cast<uint32_t>(count_image);
	info.pImageMemoryBarriers = p_image;
	vkCmdPipelineBarrier2(cmd, &info);
}
