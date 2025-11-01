#pragma once

#include <string>

class CVarParameter;

class CVarSystem
{
public:
	static CVarSystem* get();

	virtual CVarParameter* get_cvar(const std::string& str) = 0;
	virtual CVarParameter* create_int_cvar(const char* name, const char* description, int default_value, int current_value) = 0;

	virtual int* get_int_cvar(const std::string&) = 0;
	virtual void set_int_cvar(const std::string&, int value) = 0;

	virtual void draw_imgui_editor() = 0;
};

enum class CVarFlags : uint32_t
{
	None = 0,
	EditCheckbox = 1,
	EditSliderInt = 1 << 1
};

template<typename T>
struct AutoCVar
{
protected:
	int index{};
	using CVarType = T;
};

struct AutoCVar_Int : AutoCVar<int>
{
	AutoCVar_Int(const char* name, const char* description, int default_value, int current_value, CVarFlags flags);

	int get();
	void set(int value);
};