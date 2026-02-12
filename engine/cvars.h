#pragma once

#include <string>

class CVarParameter;

class CVarSystem
{
public:
	virtual ~CVarSystem() = default;
	static CVarSystem* get();

	virtual CVarParameter* get_cvar(const std::string& str) = 0;

	virtual CVarParameter* create_int_cvar(const char* name, int default_value, int current_value, int min, int max, int step_size) = 0;
	virtual int* get_int_cvar(const std::string&) = 0;
	virtual void set_int_cvar(const std::string&, int value) = 0;

	virtual CVarParameter* create_float_cvar(const char* name, float default_value, float current_value, float min, float max, float step_size) = 0;
	virtual float* get_float_cvar(const std::string&) = 0;
	virtual void set_float_cvar(const std::string&, float value) = 0;

	virtual void draw_imgui_editor() = 0;
};

enum class CVarFlags : uint32_t
{
	None = 0,
	EditCheckbox = 1,
	EditSliderInt = 1 << 1,
	EditSliderFloat = 1 << 2
};

template <typename T>
struct AutoCVar
{
protected:
	int index{};
	using CVarType = T;
};

struct AutoCVar_Int : AutoCVar<int>
{
	AutoCVar_Int(const char* name, int default_value, int current_value, CVarFlags flags, int min = 0, int max = 10, int step_size = 1);

	int get();
	void set(int value);
};

struct AutoCVar_Float : AutoCVar<float>
{
	AutoCVar_Float(const char* name, float default_value, float current_value, CVarFlags flags, float min = 0.f, float max = 1.f, float step_size = 0.1f);

	float get();
	void set(float value);
};