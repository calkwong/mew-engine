#pragma once

#include "common.h"

#include <cstdint>
#include <vector>

// TODO: private constructor and delete copy constructor/assignment
// TODO: also do the above for engine and cvars
struct TimestampManager
{
    std::vector<const char*> renderpasses{};
    std::vector<double> render_time{};
    std::vector<uint64_t> timestamps{};

    static TimestampManager& get();
    void get_query_pool_results(VkDevice device, VkQueryPool pool);
    void get_render_time(double timestamp_period);
    void add_imgui_text();
};

class ScopedTimestamp
{
public:
    ScopedTimestamp(VkCommandBuffer command_buffer, VkQueryPool query_pool, const char* renderpass);
    ~ScopedTimestamp();

private:
    VkCommandBuffer cmd{};
    VkQueryPool pool{};
    uint32_t query{};
};
