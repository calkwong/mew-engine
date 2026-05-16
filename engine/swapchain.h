#pragma once

struct Swapchain
{
    VkSwapchainKHR swapchain{};
    VkExtent2D extent{};
    bool dirty{};
    std::vector<VkImage> images{};
    std::vector<VkImageView> image_views{};
};

void create_swapchain(
    Swapchain& swapchain,
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkSurfaceKHR surface,
    uint32_t width,
    uint32_t height
);

struct SDL_Window;

void destroy_swapchain(Swapchain& swapchain, VkDevice device);
bool update_swapchain(Swapchain& swapchain, SDL_Window* window, VkPhysicalDevice physical_device, VkDevice device, VkSurfaceKHR surface);
