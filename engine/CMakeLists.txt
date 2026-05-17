add_executable(engine)

set(imgui_sources
 "${PROJECT_SOURCE_DIR}/external/imgui/imconfig.h"
 "${PROJECT_SOURCE_DIR}/external/imgui/imgui.cpp"
 "${PROJECT_SOURCE_DIR}/external/imgui/imgui.h"
 "${PROJECT_SOURCE_DIR}/external/imgui/imgui_demo.cpp"
 "${PROJECT_SOURCE_DIR}/external/imgui/imgui_draw.cpp"
 "${PROJECT_SOURCE_DIR}/external/imgui/imgui_internal.h"
 "${PROJECT_SOURCE_DIR}/external/imgui/imgui_tables.cpp"
 "${PROJECT_SOURCE_DIR}/external/imgui/imgui_widgets.cpp"
 "${PROJECT_SOURCE_DIR}/external/imgui/imstb_rectpack.h"
 "${PROJECT_SOURCE_DIR}/external/imgui/imstb_textedit.h"
 "${PROJECT_SOURCE_DIR}/external/imgui/imstb_truetype.h"
 "${PROJECT_SOURCE_DIR}/external/imgui/backends/imgui_impl_sdl3.cpp"
 "${PROJECT_SOURCE_DIR}/external/imgui/backends/imgui_impl_sdl3.h"
 "${PROJECT_SOURCE_DIR}/external/imgui/backends/imgui_impl_vulkan.cpp"
 "${PROJECT_SOURCE_DIR}/external/imgui/backends/imgui_impl_vulkan.h"
)

target_sources(engine PRIVATE
  inputs.cpp
  inputs.h
  main.cpp
  vk_engine.cpp
  vk_engine.h
  resources.cpp
  resources.h
  vk_initializers.cpp
  vk_initializers.h
  vk_loader.cpp
  vk_loader.h
  vk_pipelines.cpp
  vk_pipelines.h
  #vk_scene.cpp
  vk_scene.h
  common.h
  cvars.h
  cvars.cpp
  vk_math.h
  cache.h
  cache.cpp
  extern.cpp
  config.h
  push_constants.h
  rendergraph.h
  rendergraph.cpp
  swapchain.h
  swapchain.cpp
 ${imgui_sources}
)

#set_property(TARGET engine PROPERTY CXX_STANDARD 20)
set_target_properties(engine PROPERTIES CXX_STANDARD 20 RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_definitions(engine PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE VK_NO_PROTOTYPES IMGUI_IMPL_VULKAN_NO_PROTOTYPE)

target_include_directories(engine PUBLIC
 "${CMAKE_CURRENT_SOURCE_DIR}"
 "${PROJECT_SOURCE_DIR}/external/imgui/"
 "${PROJECT_SOURCE_DIR}/external/imgui/backends/"
)

target_link_libraries(engine PUBLIC
 vma
 glm
 # Vulkan::Vulkan
 fmt::fmt
 stb
 SDL3::SDL3
 vk-bootstrap::vk-bootstrap
 fastgltf::fastgltf
 MikkTSpace
 Tracy::TracyClient
 basis_universal
 meshoptimizer
 volk_headers
)

target_precompile_headers(engine PRIVATE
  <optional>
  <vector>
  <array>
  <memory>
  <string>
  <unordered_map>
  <functional>
  <volk.h>
  <glm/vec2.hpp>
  <glm/vec3.hpp>
  <glm/vec4.hpp>
  <glm/mat4x4.hpp>
  <glm/ext/matrix_transform.hpp>
  <glm/ext/matrix_clip_space.hpp>
  # <glm/gtx/quaternion.hpp> # does not work
)

if (WIN32)
  add_custom_command(TARGET engine POST_BUILD
 #COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:engine> $<TARGET_FILE_DIR:engine> # this is CLion on windows
 COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:engine> ${CMAKE_BINARY_DIR}/$<CONFIG>
 COMMAND_EXPAND_LISTS
)
endif()
