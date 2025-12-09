#include "camera.h"
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

glm::mat4 Camera::get_view_matrix() const
{
	// can be optimized
	glm::mat4 camera_translation = glm::translate(glm::mat4(1.0f), position);
	glm::mat4 camera_rotation = get_rotation_matrix();
	return glm::inverse(camera_translation * camera_rotation);
}

// how expensive is this?
glm::mat4 Camera::get_rotation_matrix() const
{
	glm::quat pitch_rotation{ glm::angleAxis(pitch, glm::vec3(1.0f, 0.0f, 0.0f)) };
	glm::quat yaw_rotation{ glm::angleAxis(yaw, glm::vec3(0.0f, -1.0f, 0.0f)) };

	return glm::toMat4(yaw_rotation) * glm::toMat4(pitch_rotation);
}

void Camera::process_sdl_event(SDL_Event& e)
{
	if (e.type == SDL_KEYDOWN)
	{
		if (e.key.repeat == 0 && e.key.keysym.sym == SDLK_w) { velocity.z -= 1; }
		if (e.key.repeat == 0 && e.key.keysym.sym == SDLK_s) { velocity.z += 1; }
		if (e.key.repeat == 0 && e.key.keysym.sym == SDLK_a) { velocity.x -= 1; }
		if (e.key.repeat == 0 && e.key.keysym.sym == SDLK_d) { velocity.x += 1; }
	}

	if (e.type == SDL_KEYUP)
	{
		if (e.key.keysym.sym == SDLK_w) { velocity.z += 1; }
		if (e.key.keysym.sym == SDLK_s) { velocity.z -= 1; }
		if (e.key.keysym.sym == SDLK_a) { velocity.x += 1; }
		if (e.key.keysym.sym == SDLK_d) { velocity.x -= 1; }
	}

	if (e.type == SDL_MOUSEMOTION)
	{
		yaw += static_cast<float>(e.motion.xrel) * sensitivity;
		pitch -= static_cast<float>(e.motion.yrel) * sensitivity;

		//if (pitch > 45.0f)
		//	pitch = 45.0f;
		//if (pitch < -45.0f)
		//	pitch = -45.0f; // ????
	}
}

void Camera::update(float deltatime)
{
	glm::mat4 camera_rotation = get_rotation_matrix();
	position += glm::vec3(camera_rotation * glm::vec4(velocity * speed * deltatime, 0.0f));
}

void Camera::set_perspective_matrix(float fovy, float aspect, float znear)
{
	float f = 1.0f / std::tanf(fovy / 2.0f);

	perspective = glm::mat4(
		f / aspect, 0.0f, 0.0f, 0.0f,
		0.0f, f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, -1.0f,
		0.0f, 0.0f, znear, 0.0f);
}