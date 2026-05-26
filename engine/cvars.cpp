#include "cvars.h"

#include "imgui.h"
#include <string>
#include <cassert>

CVarFlags operator|(CVarFlags a, CVarFlags b)
{
    return static_cast<CVarFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

CVarSystem* CVarSystem::get()
{
    static CVarSystem cvar_system{};
    return &cvar_system;
}

size_t CVarSystem::create_int_cvar(std::string name, std::string description, CVarFlags flags, int value, int min, int max, int step_size)
{
    if (hash.find(name) != hash.end())
        assert(0);

    auto handle = ints.size();

    ints.push_back(value);

    auto param_handle = parameters.size();
    hash[name] = param_handle;

    CVarParameter param{};
    param.type = CVarType::Int;
    param.handle = handle;
    param.description = description;
    param.flags = flags;
    param.min.i = min;
    param.max.i = max;
    param.step_size.i = step_size;

    parameters.push_back(param);
    return param_handle;
}

size_t CVarSystem::create_float_cvar(std::string name, std::string description, CVarFlags flags, float value, float min, float max, float step_size)
{
    if (hash.find(name) != hash.end())
        assert(0);

    auto handle = floats.size();

    floats.push_back(value);

    auto param_handle = parameters.size();
    hash[name] = param_handle;

    CVarParameter param{};
    param.type = CVarType::Float;
    param.handle = handle;
    param.description = description;
    param.flags = flags;
    param.min.f = min;
    param.max.f = max;
    param.step_size.f = step_size;

    parameters.push_back(param);
    return param_handle;
}

void CVarSystem::draw_imgui_editor()
{
    for (auto& p : parameters)
        edit_parameters(p);
}

void CVarSystem::edit_parameters(CVarParameter& param)
{
    const bool checkbox_flag = static_cast<uint32_t>(CVarFlags::EditCheckbox) & static_cast<uint32_t>(param.flags);
    const bool slider_int_flag = static_cast<uint32_t>(CVarFlags::EditSliderInt) & static_cast<uint32_t>(param.flags);
    const bool slider_float_flag = static_cast<uint32_t>(CVarFlags::EditDragFloat) & static_cast<uint32_t>(param.flags);
    const bool hide_flag = static_cast<uint32_t>(CVarFlags::EditHide) & static_cast<uint32_t>(param.flags);

    if (hide_flag)
        return;

    switch (param.type)
    {
    case CVarType::Int:
        if (checkbox_flag)
        {
            bool flag = ints[param.handle] == 1;
            if (ImGui::Checkbox(param.description.c_str(), &flag))
                ints[param.handle] = static_cast<int>(flag);
        }
        if (slider_int_flag)
        {
            auto* storage = &ints[param.handle];
            ImGui::SliderInt(param.description.c_str(), storage, param.min.i, param.max.i);
        }
        break;
    case CVarType::Float:
        if (slider_float_flag)
        {
            auto* storage = &floats[param.handle];
            ImGui::DragFloat(param.description.c_str(), storage, param.step_size.f, param.min.f, param.max.f);
        }
        break;
    default:
        break;
    }
}

int CVarSystem::get_int_cvar(std::string name)
{
    if (hash.find(name) == hash.end())
        assert(0);

    auto param_handle = hash[name];
    auto param = parameters[param_handle];

    auto value_handle = param.handle;

    return ints[value_handle];
}

void CVarSystem::set_int_cvar(std::string name, int value)
{
    if (hash.find(name) == hash.end())
        assert(0);

    auto param_handle = hash[name];
    auto param = parameters[param_handle];

    auto value_handle = param.handle;

    ints[value_handle] = value;
}

float CVarSystem::get_float_cvar(std::string name)
{
    if (hash.find(name) == hash.end())
        assert(0);

    auto param_handle = hash[name];
    auto param = parameters[param_handle];

    auto value_handle = param.handle;

    return floats[value_handle];
}

void CVarSystem::set_float_cvar(std::string name, float value)
{
    if (hash.find(name) == hash.end())
        assert(0);

    auto param_handle = hash[name];
    auto param = parameters[param_handle];

    auto value_handle = param.handle;

    floats[value_handle] = value;
}

AutoCVar_Int::AutoCVar_Int(
    const char* name,
    const char* description,
    CVarFlags flags,
    int value,
    int min /* = 0 */,
    int max /* = 0 */,
    int step_size /* = 0 */
)
{
    index = CVarSystem::get()->create_int_cvar(name, description, flags, value, min, max, step_size);
}

AutoCVar_Float::AutoCVar_Float(
    const char* name,
    const char* description,
    CVarFlags flags,
    float value,
    float min /* = 0 */,
    float max /* = 0 */,
    float step_size /* = 0 */
)
{
    index = CVarSystem::get()->create_float_cvar(name, description, flags, value, min, max, step_size);
}
