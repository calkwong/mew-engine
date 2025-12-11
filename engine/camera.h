#pragma once

#include "vk_math.h"
#include <SDL3/SDL_events.h>

class Camera
{
public:
	glm::vec3 position{};
	glm::vec3 velocity{};
	float speed{ 6.f };
	float sensitivity{ 0.005f };

	float pitch{ 0.0f };
	float yaw{ 0.0f };

	float near{ 50.0f };
	float far{ 0.01f };
	float fov{ 70.0f };

	glm::mat4 perspective{};

	glm::mat4 get_view_matrix() const;
	glm::mat4 get_rotation_matrix() const;
	void set_perspective_matrix(float fovy, float aspect, float znear);

	void process_sdl_event(SDL_Event& e);

	void update(float deltatime);
};