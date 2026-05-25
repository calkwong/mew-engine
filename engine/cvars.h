#pragma once

#include <string>
#include <unordered_map>

enum class CVarFlags : uint32_t
{
    None = 0,
    EditCheckbox = 1,
    EditSliderInt = 1 << 1,
    EditDragFloat = 1 << 2,
    EditHide = 1 << 3
};

CVarFlags operator|(CVarFlags a, CVarFlags b);

enum CVarType
{
    Int,
    Float,
};

struct CVarParameter
{
    CVarType type{};
    size_t handle{};
    std::string description{};
    CVarFlags flags{};

    union
    {
        int i;
        float f;
    } min;

    union
    {
        int i;
        float f;
    } max;

    union
    {
        int i;
        float f;
    } step_size;
};

class CVarSystem
{
public:
    static CVarSystem* get();

    size_t create_int_cvar(std::string name, std::string description, CVarFlags flags, int value, int min, int max, int step_size);
    size_t create_float_cvar(std::string name, std::string description, CVarFlags flags, float value, float min, float max, float step_size);

    void draw_imgui_editor();

    int get_int_cvar(std::string name);
    void set_int_cvar(std::string name, int value);
    float get_float_cvar(std::string name);
    void set_float_cvar(std::string name, float value);

    // TODO: move this to private post cleanup
    std::vector<CVarParameter> parameters{};
    std::vector<int> ints{};
    std::vector<float> floats{};

private:
    std::unordered_map<std::string, size_t> hash{};

    void edit_parameters(CVarParameter& param);
};

template<typename T>
class AutoCVar
{
protected:
    size_t index{};
};

class AutoCVar_Int : public AutoCVar<int>
{
public:
    AutoCVar_Int(const char* name, const char* description, CVarFlags flags, int value, int min = 0, int max = 0, int step_size = 0);
    int get();
};

class AutoCVar_Float : public AutoCVar<float>
{
public:
    AutoCVar_Float(const char* name, const char* description, CVarFlags flags, float value, float min = 0.0f, float max = 0.0f, float step_size = 0.0f);
    float get();
};
