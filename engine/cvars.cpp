#include "cvars.h"

#include <imgui.h>

#include <memory>
#include <string>
#include <algorithm>

enum class CVarType : char
{
	INT,
	FLOAT,
};

CVarFlags operator|(CVarFlags a, CVarFlags b)
{
	return static_cast<CVarFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

class CVarParameter
{
public:
	uint32_t array_index{};

	CVarType type{};
	CVarFlags flags{};
	std::string name{};
	std::string description{};
	union { int i; float f; } min;
	union { int i; float f; } max;
	union { int i; float f; } step_size;
};

template <typename T>
struct CVarStorage
{
	T current{};
	CVarParameter* param{};
};

template <typename T>
struct CVarArray
{
	std::unique_ptr<CVarStorage<T>[]> cvars{};
	uint32_t size{};

	CVarArray(size_t capacity)
	{
		cvars = std::make_unique<CVarStorage<T>[]>(capacity);
	}

	T get_current(int index)
	{
		return cvars[index].current;
	}

	CVarStorage<T>* get_storage_ptr(int index)
	{
		return &cvars[index];
	}

	CVarParameter* get_param(int index)
	{
		return &cvars[index].param;
	}

	void set_current(const T& value, int index)
	{
		cvars[index].current = value;
	}

	int add(const T& value, CVarParameter* param)
	{
		int index = size;

		cvars[index].current = value;
		cvars[index].param = param;

		param->array_index = index;
		size++;

		return index;
	}
};

class CVarSystemImpl final : public CVarSystem
{
public:
	constexpr static int MAX_INT_CVARS = 40;
	CVarArray<int> cvars_int{ MAX_INT_CVARS };

	constexpr static int MAX_FLOAT_CVARS = 20;
	CVarArray<float> cvars_float{ MAX_FLOAT_CVARS };

	template <typename T>
	CVarArray<T>* get_cvars_array();

	CVarParameter* get_cvar(const std::string& str) override;

	CVarParameter* create_int_cvar(const char* name, const char* description, int value, CVarFlags flags, int min, int max, int step_size) override;
	CVarParameter* create_float_cvar(const char* name, const char* description, float value, CVarFlags flags, float min, float max, float step_size) override;

	void draw_imgui_editor() override;
	void edit_parameters(CVarParameter* param);
	static CVarSystemImpl* get();

private:
	CVarParameter* init_cvar(const char* name)
	{
		if (get_cvar(name))
			return nullptr; // will fail the program

		saved_cvars[name] = CVarParameter{};

		CVarParameter& param = saved_cvars[name];

		return &param;
	}

	std::unordered_map<std::string, CVarParameter> saved_cvars{};
};

template <>
CVarArray<int>* CVarSystemImpl::get_cvars_array()
{
	return &cvars_int;
}

template <>
CVarArray<float>* CVarSystemImpl::get_cvars_array()
{
	return &cvars_float;
}

CVarParameter* CVarSystemImpl::get_cvar(const std::string& str)
{

	if (const auto it = saved_cvars.find(str); it != saved_cvars.end())
		return &(it)->second;

	return nullptr;
}

CVarSystemImpl* CVarSystemImpl::get()
{
	return static_cast<CVarSystemImpl*>(CVarSystem::get());
}

CVarSystem* CVarSystem::get()
{
	static CVarSystemImpl cvar_system{};
	return &cvar_system;
}

// if param already exists, will intentionally fail as we deference below
CVarParameter* CVarSystemImpl::create_int_cvar(const char* name, const char* description, int value, CVarFlags flags, int min, int max, int step_size)
{
	CVarParameter* param = init_cvar(name);
	if (!param)
		return nullptr;

	param->type = CVarType::INT;
	param->flags = flags;
	param->name = name;
	param->description = description;
	param->min.i = min;
	param->max.i = max;
	param->step_size.i = step_size;

	get_cvars_array<int>()->add(value, param);

	return param;
}

// will attempt to deference nullptr if cvar already exists, failing the program
AutoCVar_Int::AutoCVar_Int(const char* name, const char* description, int value, CVarFlags flags, int min, int max, int step_size)
{
	CVarParameter* param = CVarSystem::get()->create_int_cvar(name, description, value, flags, min, max, step_size);

	index = static_cast<int>(param->array_index);
}

int AutoCVar_Int::get() const
{
	return CVarSystemImpl::get()->get_cvars_array<int>()->get_current(index);
}

void AutoCVar_Int::set(int value)
{
	CVarSystemImpl::get()->get_cvars_array<int>()->set_current(value, index);
}

CVarParameter* CVarSystemImpl::create_float_cvar(const char* name, const char* description, float value, CVarFlags flags, float min, float max, float step_size)
{
	CVarParameter* param = init_cvar(name);
	if (!param)
		return nullptr;

	param->type = CVarType::FLOAT;
	param->flags = flags;
	param->name = name;
	param->description = description;
	param->min.f = min;
	param->max.f = max;
	param->step_size.f = step_size;

	get_cvars_array<float>()->add(value, param);

	return param;
}

// will attempt to deference nullptr if cvar already exists, failing the program
AutoCVar_Float::AutoCVar_Float(const char* name, const char* description, float value, CVarFlags flags, float min, float max, float step_size)
{
	CVarParameter* param = CVarSystem::get()->create_float_cvar(name, description, value, flags, min, max, step_size);

	index = static_cast<int>(param->array_index);
}

float AutoCVar_Float::get() const
{
	return CVarSystemImpl::get()->get_cvars_array<float>()->get_current(index);
}

void AutoCVar_Float::set(float value)
{
	CVarSystemImpl::get()->get_cvars_array<float>()->set_current(value, index);
}

void CVarSystemImpl::draw_imgui_editor()
{
	ImGui::Begin("Settings");

	if (ImGui::TreeNodeEx("Render", ImGuiTreeNodeFlags_DefaultOpen))
	{
		edit_parameters(get_cvar("render.vbuffer"));
		edit_parameters(get_cvar("render.mesh_shaders"));
		edit_parameters(get_cvar("render.alphaclip"));
		edit_parameters(get_cvar("render.transparent"));
		edit_parameters(get_cvar("render.point_lights"));
		edit_parameters(get_cvar("render.occlusion_cull"));
		edit_parameters(get_cvar("render.lod"));
		edit_parameters(get_cvar("render.shadows"));
		edit_parameters(get_cvar("render.taa"));
		edit_parameters(get_cvar("render.meshlet_contribution"));

		ImGui::TreePop();
	}

	// if (ImGui::TreeNodeEx("Shadows", ImGuiTreeNodeFlags_DefaultOpen))
	// {
	// 	edit_parameters(get_cvar("shadows.pcf"));
	// 	edit_parameters(get_cvar("shadows.cascade_split"));
	// 	edit_parameters(get_cvar("shadows.distance"));
	// 	edit_parameters(get_cvar("shadows.cascade_selection"));

	// 	ImGui::TreePop();
	// }

	if (ImGui::TreeNodeEx("Debug", ImGuiTreeNodeFlags_DefaultOpen))
	{
		edit_parameters(get_cvar("debug.textures"));

		ImGui::TreePop();
	}

	if (ImGui::TreeNodeEx("Misc", ImGuiTreeNodeFlags_DefaultOpen))
	{
		// edit_parameters(get_cvar("misc.draw_distance"));
		edit_parameters(get_cvar("misc.autoexposure"));
		edit_parameters(get_cvar("misc.tonemap"));
		edit_parameters(get_cvar("misc.tonemap_func"));
		edit_parameters(get_cvar("misc.freeze_camera"));

		ImGui::TreePop();
	}

	if (ImGui::TreeNodeEx("TAA", ImGuiTreeNodeFlags_DefaultOpen))
	{
		edit_parameters(get_cvar("taa.variance_clip"));
		edit_parameters(get_cvar("taa.catmull_rom"));
		edit_parameters(get_cvar("taa.mitchell"));
		edit_parameters(get_cvar("taa.ycogy"));
		edit_parameters(get_cvar("taa.dynamic"));

		ImGui::TreePop();
	}
	//
	// if (ImGui::CollapsingHeader("PBR"))
	// {
	//
	// }

	/*
	std::vector<CVarParameter*> params{};

	for (uint32_t i = 0; i < get_cvars_array<int>()->size; i++)
	{
		const auto p = get_cvars_array<int>()->get_storage_ptr(static_cast<int>(i));
		params.push_back(p->param);
	}

	for (uint32_t i = 0; i < get_cvars_array<float>()->size; i++)
	{
		const auto p = get_cvars_array<float>()->get_storage_ptr(static_cast<int>(i));
		params.push_back(p->param);
	}

	for (auto& p : params)
	{
		edit_parameters(p);
	}
	*/

	ImGui::End();
}

void CVarSystemImpl::edit_parameters(CVarParameter* param)
{
	if (!param)
		return;

	const bool checkbox_flag = static_cast<uint32_t>(CVarFlags::EditCheckbox) & static_cast<uint32_t>(param->flags);
	const bool slider_int_flag = static_cast<uint32_t>(CVarFlags::EditSliderInt) & static_cast<uint32_t>(param->flags);
	const bool slider_float_flag = static_cast<uint32_t>(CVarFlags::EditDragFloat) & static_cast<uint32_t>(param->flags);
	const bool hide_flag = static_cast<uint32_t>(CVarFlags::EditHide) & static_cast<uint32_t>(param->flags);

	if (hide_flag)
		return;

	switch (param->type)
	{
	case CVarType::INT:
		if (checkbox_flag)
		{
			bool flag = get_cvars_array<int>()->get_current(static_cast<int>(param->array_index)) == 1;
			if (ImGui::Checkbox(param->description.c_str(), &flag))
			{
				get_cvars_array<int>()->set_current(flag, static_cast<int>(param->array_index));
			}
		}
		if (slider_int_flag)
		{
			auto storage = get_cvars_array<int>()->get_storage_ptr(static_cast<int>(param->array_index));
			if (ImGui::SliderInt(param->description.c_str(), &storage->current, param->min.i, param->max.i))
			{
				get_cvars_array<int>()->set_current(storage->current, static_cast<int>(param->array_index));
			}
		}
		break;
	case CVarType::FLOAT:
		if (checkbox_flag)
		{
			bool flag = get_cvars_array<float>()->get_current(static_cast<int>(param->array_index)) == 1;
			if (ImGui::Checkbox(param->description.c_str(), &flag))
			{
				get_cvars_array<float>()->set_current(flag, static_cast<int>(param->array_index));
			}
		}
		if (slider_float_flag)
		{
			auto storage = get_cvars_array<float>()->get_storage_ptr(param->array_index);
			if (ImGui::DragFloat(param->description.c_str(), &storage->current, param->step_size.f, param->min.f, param->max.f))
			{
				get_cvars_array<float>()->set_current(storage->current, static_cast<int>(param->array_index));
			}
		}
		break;
	default:
		break;
	}
}
