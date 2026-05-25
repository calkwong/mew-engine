#include "inputs.h"
#include "vk_math.h"
#include "cvars.h"

#include <SDL3/SDL_events.h>

#include <cmath>

namespace
{
glm::vec3 rotate_quat(glm::vec3 v, glm::quat quat)
{
    glm::vec3 q = glm::vec3(quat.x, quat.y, quat.z);
    return v + glm::vec3(2.0) * cross(q, cross(q, v) + quat.w * v);
}
} // namespace

glm::mat4 Camera::get_view_matrix() const
{
    glm::quat inv_rot = glm::conjugate(get_rotation_matrix());
    glm::mat4 view = glm::mat4_cast(inv_rot);

    view[3] = glm::vec4(rotate_quat(-position, inv_rot), 1.0);

    return view;
}

glm::quat Camera::get_rotation_matrix() const
{
    const glm::quat pitch_rotation = glm::angleAxis(pitch, glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::quat yaw_rotation = glm::angleAxis(yaw, glm::vec3(0.0f, -1.0f, 0.0f));

    return yaw_rotation * pitch_rotation;
}

void Camera::process_sdl_event(const SDL_Event& e)
{
    const bool* state = SDL_GetKeyboardState(NULL);
    velocity.x = static_cast<float>(state[SDL_SCANCODE_D]) - static_cast<float>(state[SDL_SCANCODE_A]);
    velocity.z = static_cast<float>(state[SDL_SCANCODE_S]) - static_cast<float>(state[SDL_SCANCODE_W]);

    if (e.type == SDL_EVENT_MOUSE_MOTION)
    {
        // TODO: limit vertical camera rotation
        yaw += static_cast<float>(e.motion.xrel) * sensitivity;
        pitch -= static_cast<float>(e.motion.yrel) * sensitivity;
    }
}

void Camera::update(float deltatime)
{
    glm::quat camera_rotation = get_rotation_matrix();
    position += rotate_quat(glm::vec3(velocity * speed * deltatime), camera_rotation);
}

void Camera::set_perspective_matrix(float fovy, float aspect, float znear)
{
    const float f = 1.0f / std::tan(fovy / 2.0f);

    // clang-format off
	perspective = glm::mat4(
	    f / aspect, 0.0f, 0.0f, 0.0f,
	    0.0f, f, 0.0f, 0.0f,
	    0.0f, 0.0f, 0.0f, -1.0f,
	    0.0f, 0.0f, znear, 0.0f
	);
    // clang-format on
}

void toggle_cvar(const std::string& name)
{
    if (CVarSystem::get()->get_int_cvar(name) == 1)
        CVarSystem::get()->set_int_cvar(name, 0);
    else
        CVarSystem::get()->set_int_cvar(name, 1);
}

void key_callback(SDL_Window* window, SDL_Event& e)
{
    if (e.type == SDL_EVENT_KEY_DOWN)
    {
        if (e.key.repeat == 0 && e.key.key == SDLK_SPACE)
        {
            if (CVarSystem::get()->get_int_cvar("disable_camera") == 1)
            {
                CVarSystem::get()->set_int_cvar("disable_camera", 0);
                SDL_SetWindowRelativeMouseMode(window, true);
            }
            else
            {
                CVarSystem::get()->set_int_cvar("disable_camera", 1);
                SDL_SetWindowRelativeMouseMode(window, false);
            }
        }

        if (e.key.repeat == 0 && e.key.key == SDLK_R)
            toggle_cvar("imgui");

        if (e.key.repeat == 0 && e.key.key == SDLK_Z)
            toggle_cvar("taa.variance_clip");

        if (e.key.repeat == 0 && e.key.key == SDLK_J)
            toggle_cvar("taa.catmull_rom");

        if (e.key.repeat == 0 && e.key.key == SDLK_C)
            toggle_cvar("taa.ycocg");

        if (e.key.repeat == 0 && e.key.key == SDLK_T)
            toggle_cvar("taa");

        if (e.key.repeat == 0 && e.key.key == SDLK_G)
            toggle_cvar("hiz_spd");

        if (e.key.repeat == 0 && e.key.key == SDLK_F)
            toggle_cvar("shadows_rt");

        if (e.key.repeat == 0 && e.key.key == SDLK_X)
            toggle_cvar("rt");

        if (e.key.repeat == 0 && e.key.key == SDLK_Y)
            CVarSystem::get()->set_int_cvar("hot_reload", 1);
    }
}
