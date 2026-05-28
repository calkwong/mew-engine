add_executable(engine)

target_sources(engine PRIVATE
  inputs.cpp
  main.cpp
  vk_engine.cpp
  resources.cpp
  vk_loader.cpp
  pipelines.cpp
  cvars.cpp
  cache.cpp
  extern.cpp
  rendergraph.cpp
  swapchain.cpp
  descriptors.cpp
  ${PROJECT_SOURCE_DIR}/external/imgui/imgui.cpp
  ${PROJECT_SOURCE_DIR}/external/imgui/imgui_demo.cpp
  ${PROJECT_SOURCE_DIR}/external/imgui/imgui_draw.cpp
  ${PROJECT_SOURCE_DIR}/external/imgui/imgui_tables.cpp
  ${PROJECT_SOURCE_DIR}/external/imgui/imgui_widgets.cpp
  ${PROJECT_SOURCE_DIR}/external/imgui/backends/imgui_impl_sdl3.cpp
  ${PROJECT_SOURCE_DIR}/external/imgui/backends/imgui_impl_vulkan.cpp
)

add_library(basis_universal)
add_library(MikkTSpace)

target_sources(MikkTSpace PRIVATE
  ${PROJECT_SOURCE_DIR}/external/MikkTSpace/mikktspace.c
)

target_sources(basis_universal PRIVATE
  ${PROJECT_SOURCE_DIR}/external/basis_universal/transcoder/basisu_transcoder.cpp
  ${PROJECT_SOURCE_DIR}/external/basis_universal/zstd/zstd.c
)

target_include_directories(engine SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include)
target_include_directories(engine PRIVATE ${PROJECT_SOURCE_DIR}/external/glm)
target_include_directories(engine PRIVATE ${PROJECT_SOURCE_DIR}/external/stb)
target_include_directories(engine PRIVATE ${PROJECT_SOURCE_DIR}/external/MikkTSpace)
target_include_directories(engine PRIVATE ${PROJECT_SOURCE_DIR}/external/basis_universal/transcoder)
target_include_directories(engine PRIVATE ${PROJECT_SOURCE_DIR}/external/basis_universal/zstd)
target_include_directories(engine PRIVATE ${PROJECT_SOURCE_DIR}/external/imgui/ ${PROJECT_SOURCE_DIR}/external/imgui/backends/)

set_target_properties(engine PROPERTIES CXX_STANDARD 20 RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_definitions(engine PRIVATE GLM_FORCE_DEPTH_ZERO_TO_ONE VK_NO_PROTOTYPES IMGUI_IMPL_VULKAN_NO_PROTOTYPE)

find_package(TBB)
if (TBB_FOUND)
  target_link_libraries(engine PRIVATE TBB::tbb)
endif()

target_link_libraries(engine PRIVATE
 fastgltf::fastgltf
 fmt::fmt
 SDL3::SDL3
 vk-bootstrap::vk-bootstrap
 volk
 meshoptimizer
 MikkTSpace
 basis_universal
 # Tracy::TracyClient
)

target_precompile_headers(engine PRIVATE
  <optional>
  <execution>
  <algorithm>
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
)

# TODO: WIN32 specific defines, NOMINMAX etc.

if (WIN32)
  add_custom_command(TARGET engine POST_BUILD
  #COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:engine> $<TARGET_FILE_DIR:engine> # CLion
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:engine> ${CMAKE_BINARY_DIR}/$<CONFIG>
  COMMAND_EXPAND_LISTS
)
endif()
