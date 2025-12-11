#include "cvars.h"

#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_vulkan.h>

#include <memory>
#include <string>


enum class CVarType : char
{
	INT,
	//FLOAT,
};

class CVarParameter
{
public:
	friend class CVarSystemImpl;

	uint32_t array_index{};

	CVarType type{};
	CVarFlags flags{}; 
	std::string name{};
};

template<typename T>
struct CVarStorage
{
	T initial{}; // keep or remove?
	T current{};
	T min{};	   // TODO: may not be applicable for some types - reevaluate in the future
	T max{};	   // TODO: may not be applicable for some types - reevaluate in the future
	T step_size{}; // TODO: may not be applicable for some types - reevaluate in the future
	CVarParameter* param{};
};

template<typename T>
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

	T* get_current_ptr(int index)
	{
		return &cvars[index].current;
	}

	CVarStorage<T>* get_current_storage(int index)
	{
		return &cvars[index];
	}

	void set_current(const T& value, int index)
	{
		cvars[index].current = value;
	}

	template<typename T>
	int add(const T& default_value, const T& current_value, CVarParameter* param, T min, T max, T step_size)
	{
		int index = size;

		cvars[index].initial = default_value;
		cvars[index].current = current_value;
		cvars[index].param = param;
		cvars[index].min = min;
		cvars[index].max = max;
		cvars[index].step_size = step_size;

		param->array_index = index;
		size++;

		return index;
	}

	int add(const int& default_value, const int& current_value, CVarParameter* param, int min, int max, int step_size)
	{
		return add<int>(default_value, current_value, param, min, max, step_size);
	}
};


class CVarSystemImpl final : public CVarSystem
{
public:
	constexpr static int MAX_INT_CVARS = 20;
	CVarArray<int> cvars_int{ 20 };

	template<typename T>
	CVarArray<T>* get_cvars_array();

	template<>
	CVarArray<int>* get_cvars_array()
	{
		return &cvars_int;
	}

	CVarParameter* get_cvar(const std::string& str) override final;
	CVarParameter* create_int_cvar(const char* name, int default_value, int current_value, int min, int max, int step_size) override final;
	int* get_int_cvar(const std::string&) override final;
	void set_int_cvar(const std::string&, int value) override final;
	void draw_imgui_editor() override final;
	void edit_parameters(CVarParameter* param);
	static CVarSystemImpl* get();

private:
	CVarParameter* init_cvar(const char* name)
	{
		if (get_cvar(name))
			return nullptr; // will fail the program

		saved_cvars[name] = CVarParameter{};

		CVarParameter& param = saved_cvars[name];

		param.name = name;

		return &param;
	}

	std::unordered_map<std::string, CVarParameter> saved_cvars{};
};

CVarParameter* CVarSystemImpl::get_cvar(const std::string& str) 
{
	auto it = saved_cvars.find(str);

	if (it != saved_cvars.end())
		return &(*it).second;

	return nullptr;
}

CVarParameter* CVarSystemImpl::create_int_cvar(const char* name, int default_value, int current_value, int min, int max, int step_size)
{
	CVarParameter* param = init_cvar(name);
	if (!param)
		return nullptr; // param already exists

	param->type = CVarType::INT;

	get_cvars_array<int>()->add(default_value, current_value, param, min, max, step_size);

	return param;
}

int* CVarSystemImpl::get_int_cvar(const std::string& name)
{
	CVarParameter* param = get_cvar(name);
	if (!param)
		return nullptr;

	int* value = get_cvars_array<int>()->get_current_ptr(param->array_index);
	return value;
}

void CVarSystemImpl::set_int_cvar(const std::string& name, int value)
{
	CVarParameter* param = get_cvar(name);
	if (!param)
		return; // do nothing 

	get_cvars_array<int>()->set_current(value, param->array_index);
}

void CVarSystemImpl::draw_imgui_editor()
{
	ImGui::Begin("Console Variables");

	std::vector<CVarParameter*> params{};

	for (uint32_t i = 0; i < get_cvars_array<int>()->size; i++)
	{
		auto p = get_cvars_array<int>()->get_current_storage(i);
		params.push_back(p->param);
	}

	for (auto& p : params)
	{
		edit_parameters(p);
	}

	ImGui::End();
}

void CVarSystemImpl::edit_parameters(CVarParameter* param)
{
	const bool checkbox_flag = static_cast<uint32_t>(CVarFlags::EditCheckbox) & static_cast<uint32_t>(param->flags);
	const bool slider_int_flag = static_cast<uint32_t>(CVarFlags::EditSliderInt) & static_cast<uint32_t>(param->flags);

	switch (param->type)
	{
	case CVarType::INT:
		if (checkbox_flag)
		{
			bool flag = get_cvars_array<int>()->get_current(param->array_index) == 1;
			if (ImGui::Checkbox(param->name.c_str(), &flag))
			{
				get_cvars_array<int>()->set_current(static_cast<uint32_t>(flag), param->array_index);
			}
		}
		if (slider_int_flag)
		{
			auto storage = get_cvars_array<int>()->get_current_storage(param->array_index);
			if (ImGui::SliderInt(param->name.c_str(), &storage->current, storage->min, storage->max)) // TODO: clean this up, make range a variable
			{
				get_cvars_array<int>()->set_current(storage->current, param->array_index);
			}
		}
		break;

	default:
		break;
	}
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

AutoCVar_Int::AutoCVar_Int(const char* name, int default_value, int current_value, CVarFlags flags, int min, int max, int step_size)
{
	CVarParameter* param = CVarSystem::get()->create_int_cvar(name, default_value, current_value, min, max, step_size);

	param->flags = flags; // fail the program
	index = param->array_index;
}

int AutoCVar_Int::get()
{
	int value = CVarSystemImpl::get()->get_cvars_array<int>()->get_current(index);
	
	return value;
}

void AutoCVar_Int::set(int value)
{
	CVarSystemImpl::get()->get_cvars_array<int>()->set_current(value, index);
}