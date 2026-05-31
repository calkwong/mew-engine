#include "timestamps.h"
#include "common.h"

#include <imgui.h>

#include <cstdint>
#include <string>
#include <vector>

TimestampManager& TimestampManager::get()
{
    static TimestampManager manager{};
    return manager;
}

// applies lerp to a recorded timestamp; also updates the timer from outside the timestamp manager
void TimestampManager::lerp_timestamp(uint32_t current_frame, const std::string& pass, double& timer, double factor /* = 0.95 */)
{
    auto index = current_frame % 2;
    auto& frame = frames[index];

    for (uint32_t i = 0; i < frame.renderpasses.size(); i++)
    {
        // TODO: verify there's no implemention defined shenanigans here
        if (frame.renderpasses[i] == pass)
        {
            auto& recorded_time = frame.render_time[i];
            recorded_time = recorded_time + factor * (timer - recorded_time);
            timer = recorded_time;
            return;
        }
    }
}

void TimestampManager::get_render_time(uint32_t current_frame, double timestamp_period)
{
    auto index = current_frame % 2;
    auto& frame = frames[index];

    if (frame.skip)
        return;

    frame.render_time.resize(frame.renderpasses.size());

    // hardcoded lerp for GPU render time
    frame.render_time[0] = static_cast<double>(frame.timestamps[1] - frame.timestamps[0]) * timestamp_period * 1e-6;

    for (uint32_t i = 1; i < frame.render_time.size(); i++)
    {
        frame.render_time[i] = static_cast<double>(frame.timestamps[i * 2 + 1] - frame.timestamps[i * 2]) * timestamp_period * 1e-6;
    }
}

void TimestampManager::reset(uint32_t frame_number)
{
    auto& frame = frames[frame_number % 2];
    frame.timestamps.clear();
    frame.render_time.clear();
    frame.renderpasses.clear();
}

void TimestampManager::add_imgui_text(uint32_t current_frame)
{
    auto index = current_frame % 2;
    auto& frame = frames[index];
    if (frame.skip)
    {
        frame.skip = false;
        return;
    }

    for (uint32_t i = 0; i < frame.renderpasses.size(); i++)
    {
        ImGui::Text("%s", frame.renderpasses[i].c_str());
        ImGui::SameLine();
        ImGui::SetCursorPosX(300.0f);
        ImGui::Text("%.3f ms", frame.render_time[i]);
    }
}

void TimestampManager::get_query_pool_results(uint32_t current_frame, VkDevice device, VkQueryPool pool)
{
    auto index = current_frame % 2;
    auto& frame = frames[index];

    if (frame.skip)
        return;

    frame.timestamps.resize(frame.renderpasses.size() * 2);

    vkGetQueryPoolResults(
        device,
        pool,
        0,
        static_cast<uint32_t>(frame.timestamps.size()),
        frame.timestamps.size() * sizeof(uint64_t),
        frame.timestamps.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT
    );
}

uint32_t TimestampManager::add_pass(uint32_t current_frame, const std::string& pass)
{
    auto index = current_frame % 2;
    auto& frame = frames[index];
    auto size = frame.renderpasses.size();
    frame.renderpasses.push_back(pass);

    return size;
}

ScopedTimestamp::ScopedTimestamp(uint32_t current_frame, VkCommandBuffer command_buffer, VkQueryPool query_pool, const std::string& renderpass)
{
    TimestampManager& manager = TimestampManager::get();

    cmd = command_buffer;
    pool = query_pool;
    // double the returned size as we write timestamp begin and end
    query = manager.add_pass(current_frame, renderpass) * 2;

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, query);
}

ScopedTimestamp::~ScopedTimestamp()
{
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, query + 1);
}
