#pragma once

#include <volk.h>
#include <fmt/core.h>
#include <vulkan/vk_enum_string_helper.h> // string_VkResult

#define VK_CHECK(x) \
    do \
    { \
        VkResult err = x; \
        if (err) \
        { \
            fmt::println("Detected Vulkan error: {}", string_VkResult(err)); \
            abort(); \
        } \
    } while (0)
