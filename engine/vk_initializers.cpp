#include "vk_initializers.h"
#include "common.h"

#include <cstdint>

VkCommandPoolCreateInfo vkinit::command_pool_create_info(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags /*= 0*/)
{
    VkCommandPoolCreateInfo info{};

    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.queueFamilyIndex = queueFamilyIndex;
    info.flags = flags;

    return info;
}

VkCommandBufferAllocateInfo vkinit::command_buffer_allocate_info(VkCommandPool pool, uint32_t count /*= 1*/)
{
    VkCommandBufferAllocateInfo info{};

    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = pool;
    info.commandBufferCount = count;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    return info;
}

VkFenceCreateInfo vkinit::fence_create_info(VkFenceCreateFlags flags /*= 0*/)
{
    VkFenceCreateInfo info{};

    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    info.flags = flags;

    return info;
}

VkSemaphoreCreateInfo vkinit::semaphore_create_info(VkSemaphoreCreateFlags flags /*= 0*/)
{
    VkSemaphoreCreateInfo info{};

    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    info.flags = flags;

    return info;
}

VkCommandBufferBeginInfo vkinit::command_buffer_begin_info(VkCommandBufferUsageFlags flags /*= 0*/)
{
    VkCommandBufferBeginInfo info{};

    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags = flags;

    return info;
}

VkImageSubresourceRange vkinit::image_subresource_range(VkImageAspectFlags aspect_mask)
{
    VkImageSubresourceRange sub_image{};

    sub_image.aspectMask = aspect_mask;
    sub_image.baseMipLevel = 0;
    sub_image.levelCount = VK_REMAINING_MIP_LEVELS;
    sub_image.baseArrayLayer = 0;
    sub_image.layerCount = VK_REMAINING_ARRAY_LAYERS;

    return sub_image;
}

VkSemaphoreSubmitInfo vkinit::semaphore_submit_info(VkPipelineStageFlags2 stage_mask, VkSemaphore semaphore)
{
    VkSemaphoreSubmitInfo info{};

    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    info.semaphore = semaphore;
    info.value = 1;
    info.stageMask = stage_mask;
    info.deviceIndex = 0; // ?

    return info;
}

VkCommandBufferSubmitInfo vkinit::command_buffer_submit_info(VkCommandBuffer cmd)
{
    VkCommandBufferSubmitInfo info{};

    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    info.commandBuffer = cmd;
    info.deviceMask = 0; // ?

    return info;
}

VkSubmitInfo2 vkinit::submit_info(
    const VkCommandBufferSubmitInfo* submit_cmd_info,
    const VkSemaphoreSubmitInfo* signal_semaphore_info,
    const VkSemaphoreSubmitInfo* wait_semaphore_info
)
{
    VkSubmitInfo2 info{};

    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    info.pSignalSemaphoreInfos = signal_semaphore_info;
    info.signalSemaphoreInfoCount = signal_semaphore_info == nullptr ? 0 : 1;
    info.pWaitSemaphoreInfos = wait_semaphore_info;
    info.waitSemaphoreInfoCount = wait_semaphore_info == nullptr ? 0 : 1;
    info.pCommandBufferInfos = submit_cmd_info;
    info.commandBufferInfoCount = 1;

    return info;
}

VkImageCreateInfo vkinit::image_create_info(VkFormat format, VkImageUsageFlags flags, VkExtent3D extent)
{
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = extent;
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT; //
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = flags;

    return info;
}

VkImageViewCreateInfo vkinit::imageview_create_info(VkFormat format, VkImage image, VkImageAspectFlags aspect_mask)
{
    VkImageViewCreateInfo info{};

    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = format;

    info.subresourceRange = image_subresource_range(aspect_mask); // uses remaining layer and levels

    return info;
}

VkRenderingAttachmentInfo vkinit::attachment_info(VkImageView view, const VkClearValue* clear)
{
    VkRenderingAttachmentInfo info{};

    info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    info.imageView = view;
    info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    info.loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    if (clear)
        info.clearValue = *clear;

    return info;
}

VkRenderingAttachmentInfo vkinit::depth_attachment_info(VkImageView view)
{
    VkRenderingAttachmentInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    info.imageView = view;
    info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // depth pyramid?
    info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    info.clearValue.depthStencil.depth = 0.f;

    return info;
}

VkPipelineShaderStageCreateInfo vkinit::pipeline_shader_stage_create_info(
    VkShaderStageFlagBits stage,
    VkShaderModule shader_module,
    const char* entry /*= "main"*/
)
{
    VkPipelineShaderStageCreateInfo info{};

    info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage = stage;
    info.module = shader_module;
    info.pName = entry;

    return info;
}

VkRenderingInfo vkinit::rendering_info(
    VkExtent2D extent,
    const VkRenderingAttachmentInfo* color_attachment,
    const VkRenderingAttachmentInfo* depth_attachment
)
{
    VkRenderingInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    info.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, extent };
    info.layerCount = 1;
    info.colorAttachmentCount = color_attachment == nullptr ? 0 : 1;
    info.pColorAttachments = color_attachment;
    info.pDepthAttachment = depth_attachment;
    info.pStencilAttachment = nullptr;

    return info;
}
