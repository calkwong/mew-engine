add_library(glm INTERFACE)
add_library(vma INTERFACE)
add_library(stb INTERFACE)
add_library(mikktspace STATIC)
add_library(basis_universal STATIC)
add_library(meshoptimizer STATIC)

add_subdirectory(fastgltf)
add_subdirectory(fmt EXCLUDE_FROM_ALL)
add_subdirectory(SDL EXCLUDE_FROM_ALL)

add_subdirectory(vk-bootstrap)
target_include_directories(vk-bootstrap PUBLIC vk-bootstrap)

target_include_directories(vma INTERFACE VulkanMemoryAllocator/include)

target_include_directories(glm INTERFACE glm)

target_include_directories(stb INTERFACE stb)

# mikktspace
target_include_directories(mikktspace PUBLIC mikktspace)
target_sources(mikktspace PRIVATE
	mikktspace/mikktspace.c
	mikktspace/mikktspace.h
	)

# tracy
option(TRACY_ENABLE "" ON)
option(TRACY_ON_DEMAND "" ON)
add_subdirectory(tracy)

# basis_universal
target_include_directories(basis_universal PUBLIC basis_universal/transcoder basis_universal/zstd)
target_sources(basis_universal PRIVATE
	basis_universal/transcoder/basisu.h
	basis_universal/transcoder/basisu_astc_hdr_core.h
	basis_universal/transcoder/basisu_astc_helpers.h
	basis_universal/transcoder/basisu_containers.h
	basis_universal/transcoder/basisu_containers_impl.h
	basis_universal/transcoder/basisu_file_headers.h
	basis_universal/transcoder/basisu_transcoder.cpp
	basis_universal/transcoder/basisu_transcoder.h
	basis_universal/transcoder/basisu_transcoder_internal.h
	basis_universal/transcoder/basisu_transcoder_uastc.h
	basis_universal/zstd/zstddeclib.c
	basis_universal/zstd/zstd.c
	basis_universal/zstd/zstd.h
	)
	
# meshoptimizer
target_include_directories(meshoptimizer PUBLIC meshoptimizer/src)
target_sources(meshoptimizer PRIVATE 
  meshoptimizer/src/meshoptimizer.h
  meshoptimizer/src/allocator.cpp
  meshoptimizer/src/clusterizer.cpp
  meshoptimizer/src/indexanalyzer.cpp
  meshoptimizer/src/indexcodec.cpp
  meshoptimizer/src/indexgenerator.cpp
  meshoptimizer/src/overdrawoptimizer.cpp
  meshoptimizer/src/partition.cpp
  meshoptimizer/src/quantization.cpp
  meshoptimizer/src/rasterizer.cpp
  meshoptimizer/src/simplifier.cpp
  meshoptimizer/src/spatialorder.cpp
  #meshoptimizer/src/stripifier.cpp
  meshoptimizer/src/vcacheoptimizer.cpp
  meshoptimizer/src/vertexcodec.cpp
  #meshoptimizer/src/vertexfilter.cpp
  meshoptimizer/src/vfetchoptimizer.cpp
  )

# volk
add_subdirectory(volk)

