# find_package(Vulkan REQUIRED) 

add_library(vkbootstrap STATIC)
add_library(glm INTERFACE)
add_library(vma INTERFACE)
add_library(stb_image INTERFACE)
add_library(mikktspace STATIC)

add_subdirectory(fastgltf)
add_subdirectory(fmt EXCLUDE_FROM_ALL)
add_subdirectory(SDL EXCLUDE_FROM_ALL)

target_sources(vkbootstrap PRIVATE 
  vkbootstrap/VkBootstrap.h
  vkbootstrap/VkBootstrap.cpp
  )
target_include_directories(vkbootstrap PUBLIC vkbootstrap)
target_link_libraries(vkbootstrap PUBLIC Vulkan::Vulkan $<$<BOOL:UNIX>:${CMAKE_DL_LIBS}>)
set_property(TARGET vkbootstrap PROPERTY CXX_STANDARD 20)

target_include_directories(vma INTERFACE vma)

target_include_directories(glm INTERFACE glm)

target_include_directories(stb_image INTERFACE stb_image)

target_include_directories(mikktspace PUBLIC mikktspace)
target_sources(mikktspace PRIVATE
	mikktspace/mikktspace.c
	mikktspace/mikktspace.h
	)

option(TRACY_ENABLE "" ON) 
option(TRACY_ON_DEMAND "" ON)
add_subdirectory(tracy) # v0.12.2