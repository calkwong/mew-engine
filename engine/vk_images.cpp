#include "vk_images.h"
#include "vk_initializers.h"

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

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
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
				blit_info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				blit_info.srcImage = image;
				blit_info.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
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