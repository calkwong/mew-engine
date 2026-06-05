#include "queries.h"
#include "common.h"
#include "config.h"

#include <imgui.h>

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

// applies lerp to a recorded timestamp; also updates the timer from outside the timestamp manager
void TimestampManager::lerp_timestamp(uint32_t current_frame, const std::string& pass, double& timer, double factor /* = 0.95 */)
{
    auto& frame = frames[current_frame % MAX_FRAMES_IN_FLIGHT];

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
    auto& frame = frames[current_frame % MAX_FRAMES_IN_FLIGHT];

    if (frame.renderpasses.size() == 0)
        return;

    frame.render_time.resize(frame.renderpasses.size());

    for (uint32_t i = 0; i < frame.render_time.size(); i++)
    {
        frame.render_time[i] = static_cast<double>(frame.timestamps[i * 2 + 1] - frame.timestamps[i * 2]) * timestamp_period * 1e-6;
    }
}

void TimestampManager::reset(uint32_t frame_number)
{
    auto& frame = frames[frame_number % MAX_FRAMES_IN_FLIGHT];
    frame.timestamps.clear();
    frame.render_time.clear();
    frame.renderpasses.clear();
}

void TimestampManager::add_imgui_text(uint32_t current_frame)
{
    auto& frame = frames[current_frame % MAX_FRAMES_IN_FLIGHT];

    auto size = frame.renderpasses.size();
    if (size == 0)
        return;

    for (uint32_t i = 0; i < size; i++)
    {
        ImGui::Text("%s", frame.renderpasses[i].c_str());
        ImGui::SameLine();
        ImGui::SetCursorPosX(220.0f);
        ImGui::Text("%.3f ms", frame.render_time[i]);
    }
}

void TimestampManager::get_query_pool_results(uint32_t current_frame, VkDevice device, VkQueryPool pool)
{
    auto& frame = frames[current_frame % MAX_FRAMES_IN_FLIGHT];

    if (frame.renderpasses.size() == 0)
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
    auto& frame = frames[current_frame % MAX_FRAMES_IN_FLIGHT];
    auto size = frame.renderpasses.size();
    frame.renderpasses.push_back(pass);

    return size;
}

ScopedTimestamp::ScopedTimestamp(TimestampManager* manager, uint32_t current_frame, VkCommandBuffer command_buffer, VkQueryPool query_pool, const std::string& renderpass)
    : manager{ manager }, cmd{ command_buffer }, pool{ query_pool }
{
    // double the returned size as we write timestamp begin and end
    query = manager->add_pass(current_frame, renderpass) * 2;

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, query);
}

ScopedTimestamp::~ScopedTimestamp()
{
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, query + 1);
}

void PipelineQueryManager::get_query_pool_results(uint32_t current_frame, VkDevice device, VkQueryPool pool, PipelineQueryType type)
{
    auto& frame = frames[current_frame % MAX_FRAMES_IN_FLIGHT];

    uint64_t* data{};
    uint32_t size{};

    switch (type)
    {
    case PipelineQueryType::Vertex:
        size = frame.pipeline_results.size();
        data = frame.pipeline_results.data();
        break;
    case PipelineQueryType::Mesh:
        size = frame.mesh_pipeline_results.size();
        data = frame.mesh_pipeline_results.data();
        break;
    default:
        assert(0);
    }

    if (size == 0)
        return;

    vkGetQueryPoolResults(
        device,
        pool,
        0,
        size,
        size * sizeof(uint64_t),
        data,
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT
    );
}

void PipelineQueryManager::add_imgui_text(uint32_t current_frame)
{
    auto& frame = frames[current_frame % MAX_FRAMES_IN_FLIGHT];

    if (frame.pipeline_results.size() == 0 && frame.mesh_pipeline_results.size() == 0)
        return;

    uint64_t total{};

    for (auto result : frame.pipeline_results)
        total += result;

    for (auto result : frame.mesh_pipeline_results)
        total += result;

    {
        ImGui::Text("Triangles");
        ImGui::SameLine();
        ImGui::SetCursorPosX(220.0f);
        ImGui::Text("%u", static_cast<unsigned int>(total));

        ImGui::Text("Triangles");
        ImGui::SameLine();
        ImGui::SetCursorPosX(220.0f);
        ImGui::Text("%.1fM", static_cast<double>(total) * 1e-6);
    }
}

void PipelineQueryManager::reset(uint32_t current_frame)
{
    auto& frame = frames[current_frame % MAX_FRAMES_IN_FLIGHT];

    frame.pipeline_results.clear();
    frame.mesh_pipeline_results.clear();
}

uint32_t PipelineQueryManager::add_query(uint32_t current_frame, PipelineQueryType type)
{
    auto& frame = frames[current_frame % MAX_FRAMES_IN_FLIGHT];

    switch (type)
    {
    case PipelineQueryType::Vertex:
        frame.pipeline_results.push_back(0);
        return static_cast<uint32_t>(frame.pipeline_results.size());
    case PipelineQueryType::Mesh:
        frame.mesh_pipeline_results.push_back(0);
        return static_cast<uint32_t>(frame.mesh_pipeline_results.size());
    default:
        assert(0);
    }
}

ScopedPipelineQuery::ScopedPipelineQuery(PipelineQueryManager* manager, uint32_t current_frame, VkCommandBuffer command_buffer, VkQueryPool query_pool, PipelineQueryType type)
    : manager(manager), cmd(command_buffer), pool(query_pool)
{
    query = manager->add_query(current_frame, type);
    vkCmdBeginQuery(cmd, query_pool, query, 0);
}

ScopedPipelineQuery::~ScopedPipelineQuery()
{
    vkCmdEndQuery(cmd, pool, query);
}
