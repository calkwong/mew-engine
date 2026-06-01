#pragma once

#include "common.h"
#include "config.h"

#include <cstdint>
#include <string>
#include <vector>

class TimestampManager
{
public:
    void get_query_pool_results(uint32_t current_frame, VkDevice device, VkQueryPool pool);
    void get_render_time(uint32_t current_frame, double timestamp_period);
    void add_imgui_text(uint32_t current_frame);
    void reset(uint32_t current_frame);
    void lerp_timestamp(uint32_t current_frame, const std::string& pass, double& timer, double factor = 0.95);
    uint32_t add_pass(uint32_t current_frame, const std::string& pass);

private:
    struct Frame
    {
        std::vector<std::string> renderpasses{};
        std::vector<double> render_time{};
        std::vector<uint64_t> timestamps{};
        bool skip = true; // hack
    };

    Frame frames[MAX_FRAMES_IN_FLIGHT]{};
};

class ScopedTimestamp
{
public:
    ScopedTimestamp(TimestampManager* manager, uint32_t current_frame, VkCommandBuffer command_buffer, VkQueryPool query_pool, const std::string& renderpass);
    ~ScopedTimestamp();
    ScopedTimestamp(const ScopedTimestamp& timestamp) = delete;
    ScopedTimestamp& operator=(const ScopedTimestamp& timestamp) = delete;

private:
    TimestampManager* manager{};
    VkCommandBuffer cmd{};
    VkQueryPool pool{};
    uint32_t query{};
};
