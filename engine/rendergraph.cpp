#include "rendergraph.h"
#include "common.h"
#include "resources.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

void RenderGraph::add_resource(VkImage image)
{
    resources.emplace_back(TrackedResource{ image });
}

uint32_t RenderGraph::get_resources_size() const
{
    return static_cast<uint32_t>(resources.size());
}

uint32_t Pass::get_resource_index(const std::string& name, VkImage image)
{
    uint32_t index = -1;

    // register resource into rendergraph
    if (graph->resource_indices.find(name) == graph->resource_indices.end())
    {
        index = graph->get_resources_size();
        graph->resource_indices[name] = index;
        graph->add_resource(image);
    }
    else
        index = graph->resource_indices[name];

    return index;
}

void Pass::add_depth_stencil_output(const std::string& name, VkImage image)
{
    auto index = get_resource_index(name, image);
    // for simplicity include both early and late
    flushes.emplace_back(Barrier{ index, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT });
}

void Pass::add_color_output(const std::string& name, VkImage image)
{
    auto index = get_resource_index(name, image);
    flushes.emplace_back(Barrier{ index, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT });
}

void Pass::add_image_read(const std::string& name, VkImage image)
{
    auto index = get_resource_index(name, image);
    auto stage = pass_type == PassType::GraphicsPass ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    invalidates.emplace_back(Barrier{ index, stage });
}

void Pass::add_image_write(const std::string& name, VkImage image)
{
    auto index = get_resource_index(name, image);
    auto stage = pass_type == PassType::GraphicsPass ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    flushes.emplace_back(Barrier{ index, stage });
}

void Pass::add_storage_buffer_read(const std::string& name)
{
    auto index = get_resource_index(name);
    auto stage = pass_type == PassType::GraphicsPass ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    invalidates.emplace_back(Barrier{ index, stage });
}

void Pass::add_storage_buffer_write(const std::string& name)
{
    auto index = get_resource_index(name);
    auto stage = pass_type == PassType::GraphicsPass ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    flushes.emplace_back(Barrier{ index, stage });
}

void Pass::add_indirect_buffer_read(const std::string& name)
{
    auto index = get_resource_index(name);
    invalidates.emplace_back(Barrier{ index, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT });
}

void RenderGraph::add_pass(const std::string& name, Pass::PassType pass_type, std::function<void(Pass& pass)> setup, std::function<void()> execute)
{
    if (pass_indices.find(name) != pass_indices.end())
        assert(0);

    uint32_t index = static_cast<uint32_t>(passes.size());
    pass_indices[name] = index;

    Pass& pass = passes.emplace_back(Pass{ .graph = this, .name = name, .pass_type = pass_type, .callback = execute });

    setup(pass);
}

// No support for resource aliasing, transient resources, renderpass, pass reordering, sorting, pass/barrier merging.
void RenderGraph::bake()
{
    build_barriers();
}

// We group up all images that need UNDEFINED -> GENERAL before running any passes
// Write only images that discard between pass executions not currently supported as we don't have such cases.
// For buffers we just emit gigabarriers.
void RenderGraph::build_barriers()
{
    struct State
    {
        VkPipelineStageFlags2 read = 0;
        VkPipelineStageFlags2 write = 0;
    };

    std::vector<State> states{};
    states.reserve(resources.size());

    for (auto& pass : passes)
    {
        states.clear();
        states.resize(resources.size());

        // mark if resource is read
        for (const auto& invalidate : pass.invalidates)
        {
            auto& state = states[invalidate.index];
            state.read |= invalidate.stages;
        }

        // mark if resource is written to
        for (const auto& flush : pass.flushes)
        {
            auto& state = states[flush.index];
            state.write |= flush.stages;
        }

        for (uint32_t index = 0; index < states.size(); index++)
        {
            auto& state = states[index];

            if (state.read == 0 && state.write == 0)
                continue;

            auto& res = resources[index];

            // Only interested in write-only images that need early discard
            if (state.read == 0 && state.write != 0 && res.image != VK_NULL_HANDLE)
            {
                if (state.write & VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
                    early_depth_discards.push_back(index);
                else
                    early_discards.push_back(index);
            }
        }
    }
}

void RenderGraph::execute(VkCommandBuffer cmd)
{
    if (passes.size() <= 0)
        return;

    std::vector<VkImageMemoryBarrier2> image_memory_barriers{};
    std::vector<VkMemoryBarrier2> memory_barriers{};
    for (auto& index : early_discards)
    {
        image_memory_barriers.emplace_back(image_barrier(
            resources[index].image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
        ));
    }

    for (auto& index : early_depth_discards)
    {
        image_memory_barriers.emplace_back(image_barrier(
            resources[index].image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT
        ));
    }

    memory_barriers.emplace_back(buffer_barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT));
    pipeline_barrier(cmd, memory_barriers.data(), 1, image_memory_barriers.data(), image_memory_barriers.size());

    // TODO: make this safer
    passes[0].callback();

    // Gigabarrier everything else, this is the most efficient but for development sanity
    for (size_t i = 1; i < passes.size(); i++)
    {
        auto& pass = passes[i];
        giga_barrier(cmd);
        pass.callback();
    }

    giga_barrier(cmd);
}

void RenderGraph::print() const
{
    for (const auto& pass : passes)
    {
        fmt::println("{}", pass.name);
    }
    fmt::println("--------------");
}
