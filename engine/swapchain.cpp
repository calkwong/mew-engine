#include "common.h"
#include "swapchain.h"

#include <VkBootstrap.h>
#include <SDL3/SDL_init.h>

void create_swapchain(
    Swapchain& swapchain,
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkSurfaceKHR surface,
    uint32_t width,
    uint32_t height
)
{
    vkb::SwapchainBuilder swapchainBuilder{ physical_device, device, surface };

    vkb::Swapchain vkbSwapchain =
        swapchainBuilder
            //.use_default_format_selection()
            .set_desired_format(VkSurfaceFormatKHR{ .format = VK_FORMAT_B8G8R8A8_UNORM, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
            // .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
            .set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)
            .set_desired_extent(width, height)
            .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            .build()
            .value();

    swapchain.swapchain = vkbSwapchain.swapchain;
    swapchain.extent = vkbSwapchain.extent;
    swapchain.images = vkbSwapchain.get_images().value();
    swapchain.image_views = vkbSwapchain.get_image_views().value();
}

void destroy_swapchain(Swapchain& swapchain, VkDevice device)
{
    // destroys images held
    vkDestroySwapchainKHR(device, swapchain.swapchain, nullptr);

    for (auto& view : swapchain.image_views)
    {
        vkDestroyImageView(device, view, nullptr);
    }

    swapchain.images.clear();
    swapchain.image_views.clear();
}

bool update_swapchain(Swapchain& swapchain, SDL_Window* window, VkPhysicalDevice physical_device, VkDevice device, VkSurfaceKHR surface)
{
    // TODO: do we need to handle width == height == 0?

    int w{};
    int h{};
    SDL_GetWindowSizeInPixels(window, &w, &h);

    // overall handles lots of x11/wayland specific issues
    // also works around a possible niri + nvidia only issue,
    // see: https://github.com/niri-wm/niri/issues/2335
    if (swapchain.extent.width != w || swapchain.extent.height != h || swapchain.dirty)
    {
        vkDeviceWaitIdle(device);

        destroy_swapchain(swapchain, device);
        create_swapchain(swapchain, physical_device, device, surface, w, h);
        fmt::println("swapchain size: {}x{}", swapchain.extent.width, swapchain.extent.height);

        swapchain.dirty = false;
        return true;
    }

    return false;
}
