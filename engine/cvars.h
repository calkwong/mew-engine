#pragma once

#include <string>

class CVarParameter;

enum class CVarFlags : uint32_t
{
	None = 0,
	EditCheckbox = 1,
	EditSliderInt = 1 << 1,
	EditDragFloat = 1 << 2,
	EditHide = 1 << 3
};

CVarFlags operator|(CVarFlags a, CVarFlags b);

class CVarSystem
{
public:
	virtual ~CVarSystem() = default;
	static CVarSystem* get();

	virtual CVarParameter* get_cvar(const std::string& str) = 0;

	virtual CVarParameter* create_int_cvar(const char* name, const char* description, int current_value, CVarFlags flags, int min, int max, int step_size) = 0;
	virtual CVarParameter* create_float_cvar(const char* name, const char* description, float current_value, CVarFlags flags, float min, float max, float step_size) = 0;

	virtual void draw_imgui_editor() = 0;
};

template <typename T>
struct AutoCVar
{
protected:
	int index{};
};

struct AutoCVar_Int : AutoCVar<int>
{
	AutoCVar_Int(const char* name, const char* description, int current_value, CVarFlags flags, int min = 0, int max = 1, int step_size = 1);

	int get() const;
	void set(int value);
};

struct AutoCVar_Float : AutoCVar<float>
{
	AutoCVar_Float(const char* name, const char* description, float current_value, CVarFlags flags, float min = 0.f, float max = 1.f, float step_size = 0.05f);

	float get() const;
	void set(float value);
};

int get_int_cvars(const std::string& name);
float get_float_cvars(const std::string& name);

void set_int_cvars(const std::string& name, int value);
void set_float_cvars(const std::string& name, float value);
