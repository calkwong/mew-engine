cmake_minimum_required (VERSION 3.14)

project ("mewengine")

find_package(Vulkan REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(external)
add_subdirectory(engine)

find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.frag"
    "${PROJECT_SOURCE_DIR}/shaders/*.vert"
    "${PROJECT_SOURCE_DIR}/shaders/*.comp"
    "${PROJECT_SOURCE_DIR}/shaders/*.mesh"
    )

file(GLOB_RECURSE GLSL_HEADER_FILES
	"${PROJECT_SOURCE_DIR}/shaders/*.glsl"
    )

foreach(GLSL ${GLSL_SOURCE_FILES})
  message(STATUS "BUILDING SHADER")
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/compiled/${FILE_NAME}.spv")
  message(STATUS ${GLSL})
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV} --target-env vulkan1.4 #-gVS
    DEPENDS ${GLSL} ${GLSL_HEADER_FILES})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

# slang
# TODO:
# - consider using slang compilation API if command line does not cache modules

find_program(SLANG_COMPILER slangc HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

file(GLOB SLANG_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.slang"
    )

file(GLOB SLANG_MODULES
    "${PROJECT_SOURCE_DIR}/shaders/modules/*.slang"
    )

foreach(SLANG ${SLANG_MODULES})
  message(STATUS "BUILDING MODULES")
  get_filename_component(FILE_NAME ${SLANG} NAME)
  set(SLANG_MODULE "${PROJECT_SOURCE_DIR}/shaders/compiled/${FILE_NAME}.slang-module")
  message(STATUS ${SLANG})
  add_custom_command(
    OUTPUT ${SLANG_MODULE}
    COMMAND ${SLANG_COMPILER} ${SLANG} -o ${SLANG_MODULE}
    DEPENDS ${SLANG})
  list(APPEND SPIRV_BINARY_FILES ${SLANG_MODULE})
endforeach(SLANG)

foreach(SLANG ${SLANG_SOURCE_FILES})
  message(STATUS "BUILDING SHADER")
  get_filename_component(FILE_NAME ${SLANG} NAME)
  set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/compiled/${FILE_NAME}.spv")
  message(STATUS ${SLANG})
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${SLANG_COMPILER} ${SLANG} -target spirv -matrix-layout-column-major -fvk-use-entrypoint-name
      -I "${PROJECT_SOURCE_DIR}/shaders/compiled/" -I "${PROJECT_SOURCE_DIR}/shaders/modules/"
      -Wno-39001 -o ${SPIRV}
    DEPENDS ${SLANG} ${SLANG_MODULES})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(SLANG)

add_custom_target(
    Shaders
    DEPENDS ${SPIRV_BINARY_FILES}
    )

add_dependencies(engine Shaders)
