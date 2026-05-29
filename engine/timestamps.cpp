#include "timestamps.h"
#include "common.h"

#include <imgui.h>

#include <cstdint>
#include <vector>

TimestampManager& TimestampManager::get()
{
    static TimestampManager manager{};
    return manager;
}

void TimestampManager::get_render_time(double timestamp_period)
{
    render_time.clear();
    render_time.resize(renderpasses.size());

    for (uint32_t i = 0; i < timestamps.size(); i = i + 2)
    {
        render_time[i] = static_cast<double>(timestamps[i + 1] - timestamps[i]) * timestamp_period * 1e-6;
    }
}

void TimestampManager::add_imgui_text()
{
    for (uint32_t i = 0; i < renderpasses.size(); i++)
    {
        ImGui::Text("%s", renderpasses[i]);
        ImGui::SameLine();
        ImGui::SetCursorPosX(100.0f);
        ImGui::Text("%.3f ms", render_time[i]);
    }
}

void TimestampManager::get_query_pool_results(VkDevice device, VkQueryPool pool)
{
    timestamps.clear();
    timestamps.resize(renderpasses.size() * 2);

    vkGetQueryPoolResults(
        device,
        pool,
        0,
        static_cast<uint32_t>(timestamps.size()),
        timestamps.size() * sizeof(uint64_t),
        timestamps.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT
    );
}

ScopedTimestamp::ScopedTimestamp(VkCommandBuffer command_buffer, VkQueryPool query_pool, const char* renderpass)
{
    TimestampManager& manager = TimestampManager::get();

    cmd = command_buffer;
    pool = query_pool;
    query = static_cast<uint32_t>(manager.renderpasses.size()) * 2;

    manager.renderpasses.push_back(renderpass);

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, query);
}

ScopedTimestamp::~ScopedTimestamp()
{
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, query + 1);
}
