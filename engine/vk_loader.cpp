#include "common.h"
#include "vk_math.h"
#include "vk_loader.h"
#include "cache.h"
#include "resources.h"
#include "vk_engine.h"

#include <basisu_transcoder.h>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <fmt/core.h>
#include <meshoptimizer.h>
#include <mikktspace.h>
#include <stb_image.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <variant>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

// meshlet_indices stores meshlet vertices & triangles, meshlet stores offset into meshlet_indices, and triangle/vertices count
void optimize_mesh(
    std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, std::vector<uint32_t>& meshlet_indices, std::vector<Meshlet>& meshlets,
    GeoSurface& surface, std::vector<Vertex>& combined_vertices, std::vector<uint32_t>& combined_indices
)
{
	// indexing
	std::vector<uint32_t> remap(vertices.size());
	size_t unique_vertices = meshopt_generateVertexRemap(remap.data(), indices.data(), indices.size(), vertices.data(), vertices.size(), sizeof(Vertex));

	meshopt_remapIndexBuffer(indices.data(), indices.data(), indices.size(), remap.data());
	meshopt_remapVertexBuffer(vertices.data(), vertices.data(), vertices.size(), sizeof(Vertex), remap.data());

	vertices.resize(unique_vertices);
	size_t vertex_count = vertices.size();

	// vertex cache optimization
	meshopt_optimizeVertexCache(indices.data(), indices.data(), indices.size(), vertex_count);

	// vertex fetch optmization
	meshopt_optimizeVertexFetch(vertices.data(), indices.data(), indices.size(), vertices.data(), vertex_count, sizeof(Vertex));

	glm::vec3 center{};

	std::vector<glm::vec3> positions(vertex_count);
	for (size_t i = 0; i < vertex_count; i++)
	{
		positions[i] = vertices[i].position;
		center += positions[i];
	}

	std::vector<glm::vec3> normals(vertex_count);
	for (size_t i = 0; i < vertex_count; i++)
	{
		normals[i] = vertices[i].normal;
	}

	center /= vertices.size();
	float radius = 0.0;

	for (size_t i = 0; i < vertex_count; i++)
	{
		radius = std::max(radius, glm::distance(center, positions[i]));
	}

	surface.bounds.origin = center;
	surface.bounds.radius = radius;

	float lod_error_scale = meshopt_simplifyScale(&positions[0].x, vertex_count, sizeof(glm::vec3));
	float target_error = 1e-1f;
	float lod_error = 0.f;

	constexpr float attr_weights[3] = { 1.0f, 1.0f, 1.0f }; // for normals
	float next_error{};

	// meshlets
	constexpr size_t max_vertices = 64;
	constexpr size_t max_triangles = 124;
	constexpr float cone_weight = 0.f;
	constexpr uint32_t MAX_LOD = 8;
	float simplify_threshold = 0.6f;
	while (surface.lod_count < MAX_LOD)
	{
		uint32_t first_index = static_cast<uint32_t>(combined_indices.size());
		uint32_t count = static_cast<uint32_t>(indices.size());

		// appending mesh indices
		combined_indices.insert(combined_indices.end(), indices.begin(), indices.end());

		MeshLod lod_info{};
		lod_info.first_index = first_index;
		lod_info.count = count;
		lod_info.error = lod_error * lod_error_scale;

		size_t target_index_count = static_cast<size_t>(static_cast<float>(indices.size()) * simplify_threshold) / 3 * 3;

		size_t max_meshlets = meshopt_buildMeshletsBound(indices.size(), max_vertices, max_triangles);
		std::vector<meshopt_Meshlet> meshopt_meshlets(max_meshlets);
		std::vector<uint32_t> meshlet_vertices(max_meshlets * max_vertices); // TODO: should size be indices.size()?
		std::vector<uint8_t> meshlet_triangles(max_meshlets * max_triangles * 3); // TODO: should size be indices.size()?

		size_t meshlet_count = meshopt_buildMeshlets(meshopt_meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices.data(), indices.size(), &positions[0].x, vertex_count, sizeof(glm::vec3), max_vertices, max_triangles, cone_weight);

		// trim arrays
		// meshopt_Meshlet's triangle_offset already accounts for alignment padding
		const meshopt_Meshlet& last = meshopt_meshlets[meshlet_count - 1];
		meshlet_vertices.resize(last.vertex_offset + last.vertex_count);
		meshlet_triangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3)); // 4 byte alignment
		meshopt_meshlets.resize(meshlet_count);

		uint32_t meshlet_offset = static_cast<uint32_t>(meshlets.size());
		lod_info.meshlet_offset = meshlet_offset;
		lod_info.meshlet_count = static_cast<uint32_t>(meshlet_count);

		if (surface.lod_count == 0)
		{
			surface.meshlet_bits = static_cast<uint32_t>(meshlet_count);
		}

		surface.mesh_lods[surface.lod_count++] = lod_info;
		uint32_t meshlet_indices_offset = static_cast<uint32_t>(meshlet_indices.size());
		for (auto& m : meshopt_meshlets)
		{
			meshopt_optimizeMeshlet(&meshlet_vertices[m.vertex_offset], &meshlet_triangles[m.triangle_offset], m.triangle_count, m.vertex_count);

			meshopt_Bounds bounds = meshopt_computeMeshletBounds(&meshlet_vertices[m.vertex_offset], &meshlet_triangles[m.triangle_offset], m.triangle_count, &positions[0].x, vertex_count, sizeof(glm::vec3));

			Meshlet new_meshlet{};
			new_meshlet.center = glm::vec3(bounds.center[0], bounds.center[1], bounds.center[2]);
			new_meshlet.radius = bounds.radius;
			new_meshlet.data_offset = meshlet_indices_offset;
			new_meshlet.vertex_count = m.vertex_count;
			new_meshlet.triangle_count = m.triangle_count;

			meshlets.push_back(new_meshlet);

			meshlet_indices_offset += m.vertex_count + m.triangle_count * 3;

			uint32_t combined_vertices_count = static_cast<uint32_t>(combined_vertices.size());
			for (size_t idx = 0; idx < m.vertex_count; idx++)
			{
				meshlet_indices.push_back(meshlet_vertices[m.vertex_offset + idx] + combined_vertices_count);
			}

			for (size_t idx = 0; idx < m.triangle_count; idx++)
			{
				meshlet_indices.push_back(meshlet_triangles[m.triangle_offset + idx * 3 + 0]);
				meshlet_indices.push_back(meshlet_triangles[m.triangle_offset + idx * 3 + 1]);
				meshlet_indices.push_back(meshlet_triangles[m.triangle_offset + idx * 3 + 2]);
			}
		}

		if (surface.lod_count < MAX_LOD)
		{
			size_t new_size = meshopt_simplifyWithAttributes(indices.data(), indices.data(), indices.size(), &positions[0].x, vertex_count, sizeof(glm::vec3), &normals[0].x, sizeof(glm::vec3), &attr_weights[0], 3, nullptr, target_index_count, target_error, 0, &next_error);

			assert(new_size <= indices.size());

			if (new_size == 0)
				break;

			// discard LOD if too similar to previous LOD, saves memory
			if (new_size >= static_cast<size_t>(static_cast<float>(indices.size()) * 0.85))
				break;

			indices.resize(new_size);

			lod_error = std::max(lod_error, next_error); // accumulate error as its technically possible for lower LOD to have smaller error

			meshopt_optimizeVertexCache(indices.data(), indices.data(), new_size, vertex_count);
		}
	}
}

bool read_ktx2_file(const char* filename, std::vector<uint8_t>& ktx_data)
{
	// cursor at the end
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open())
	{
		return false;
	}

	// find what the size of the file is by looking up the location of the cursor
	// because the cursor is at the end, it gives the size directly in bytes
	const size_t file_size = file.tellg();

	// spirv expects the buffer to be on uint32, so make sure to reserve a int
	// vector big enough for the entire file
	std::vector<uint8_t> buffer(file_size);

	// put file cursor at beginning
	file.seekg(0);

	// load the entire file into the buffer
	file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(file_size));

	// now that the file is loaded into the buffer, we can close it
	file.close();

	ktx_data = std::move(buffer);

	return true;
}

AllocatedImage basisu_load(VulkanEngine* engine, const char* filepath)
{
	AllocatedImage new_image{};

	// unsigned char* data{};

	std::vector<uint8_t> buffer{};

	if (!read_ktx2_file(filepath, buffer))
	{
		assert(0);
	}

	// create the KTX2 transcoder object
	basist::ktx2_transcoder transcoder{};

	// initialize the transcoder
	if (!transcoder.init(buffer.data(), static_cast<uint32_t>(buffer.size())))
	{
		assert(0);
	}

	auto target_format = basist::transcoder_texture_format::cTFBC7_RGBA;

	VkFormat vk_format{};
	auto transfer_func = transcoder.get_dfd_transfer_func();
	switch (transfer_func)
	{
	case basist::KTX2_KHR_DF_TRANSFER_SRGB:
		vk_format = VK_FORMAT_BC7_SRGB_BLOCK;
		break;
	case basist::KTX2_KHR_DF_TRANSFER_LINEAR:
		vk_format = VK_FORMAT_BC7_UNORM_BLOCK;
		break;
	default:
		assert(0);
	}

	std::vector<basist::ktx2_image_level_info> level_infos(transcoder.get_levels());
	const uint32_t mip_level = transcoder.get_levels();

	for (uint32_t i = 0; i < mip_level; i++)
	{
		transcoder.get_image_level_info(level_infos[i], i, 0, 0);
	}

	// const uint32_t width = level_infos[0].m_orig_width;
	// const uint32_t height = level_infos[0].m_orig_height;

	const uint32_t bytes_per_block_or_pixel = basist::basis_get_bytes_per_block_or_pixel(target_format);
	uint32_t num_blocks_or_pixels = 0;
	VkDeviceSize buffer_size = 0;

	for (uint32_t i = 0; i < mip_level; i++)
	{
		num_blocks_or_pixels = level_infos[i].m_total_blocks;
		buffer_size += bytes_per_block_or_pixel * num_blocks_or_pixels;
	}

	auto header = transcoder.get_header();
	auto ss = header.m_supercompression_scheme;
	if (ss == basist::KTX2_SS_NONE)
	{
		assert(0); // TODO: transcoding not req, verify ktx2 in GPU ready format - not currently handled
	}

	transcoder.start_transcoding();

	std::vector<char> ktx_data(buffer_size);
	char* ktx_data_ptr = ktx_data.data();

	for (uint32_t i = 0; i < mip_level; i++)
	{
		num_blocks_or_pixels = level_infos[i].m_total_blocks;
		uint32_t output_size = bytes_per_block_or_pixel * num_blocks_or_pixels;
		if (!transcoder.transcode_image_level(i, 0, 0, ktx_data_ptr, output_size, target_format))
			assert(0);
		ktx_data_ptr += output_size;
	}

	AllocatedBuffer upload_buffer = create_buffer(engine->allocator, buffer_size, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

	memcpy(upload_buffer.info.pMappedData, ktx_data.data(), buffer_size);

	std::vector<VkBufferImageCopy> copy_regions{};

	uint32_t buffer_offset = 0;
	for (uint32_t i = 0; i < mip_level; i++)
	{
		num_blocks_or_pixels = level_infos[i].m_total_blocks;
		uint32_t offset = bytes_per_block_or_pixel * num_blocks_or_pixels;

		VkBufferImageCopy copy_region{};
		copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copy_region.imageSubresource.mipLevel = i;
		copy_region.imageSubresource.baseArrayLayer = 0;
		copy_region.imageSubresource.layerCount = 1;
		copy_region.imageExtent = VkExtent3D{ level_infos[i].m_orig_width, level_infos[i].m_orig_height, 1 };
		copy_region.bufferOffset = buffer_offset;
		copy_regions.push_back(copy_region);

		buffer_offset += offset;
	}

	VkExtent3D vk_extent = VkExtent3D{ level_infos[0].m_orig_width, level_infos[0].m_orig_height, 1 };
	new_image = create_image(engine->device, engine->allocator, vk_extent, vk_format, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, true);

	// clang-format off
	immediate_submit(engine->device, engine->graphics_queue, engine->imm_command_buffer, engine->imm_fence, [&](VkCommandBuffer cmd)
	    {
			vkutil::transition_image(
				cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				0,
				VK_PIPELINE_STAGE_2_TRANSFER_BIT,
				0,
				VK_ACCESS_2_TRANSFER_WRITE_BIT
			);

			vkCmdCopyBufferToImage(cmd, upload_buffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<uint32_t>(copy_regions.size()), copy_regions.data());

			vkutil::transition_image(
				cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_2_TRANSFER_BIT,
				VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
				VK_ACCESS_2_TRANSFER_WRITE_BIT,
				VK_ACCESS_2_SHADER_READ_BIT
			);
	    }
	);
	// clang-format on

	destroy_buffer(engine->allocator, upload_buffer);

	return new_image;
}

VkFilter extract_filter(fastgltf::Filter filter)
{
	switch (filter)
	{
	case fastgltf::Filter::Nearest:
	case fastgltf::Filter::NearestMipMapNearest:
	case fastgltf::Filter::NearestMipMapLinear:
		return VK_FILTER_NEAREST;

	case fastgltf::Filter::Linear:
	case fastgltf::Filter::LinearMipMapNearest:
	case fastgltf::Filter::LinearMipMapLinear:
	default:
		return VK_FILTER_LINEAR;
	}
}

VkSamplerMipmapMode extract_mipmap(fastgltf::Filter filter)
{
	switch (filter)
	{
	case fastgltf::Filter::NearestMipMapNearest:
	case fastgltf::Filter::LinearMipMapNearest:
		return VK_SAMPLER_MIPMAP_MODE_NEAREST;

	case fastgltf::Filter::NearestMipMapLinear:
	case fastgltf::Filter::LinearMipMapLinear:
	default:
		return VK_SAMPLER_MIPMAP_MODE_LINEAR;
	}
}

std::optional<AllocatedImage> load_image(VulkanEngine* engine, const std::string& parent_path, fastgltf::Asset& asset, fastgltf::Image& image, VkFormat format, bool mipmapped = false)
{
	AllocatedImage new_image{};

	int width{};
	int height{};
	int channels{};

	// clang-format off
	std::visit(
	    fastgltf::visitor{
	        [](auto& arg) {},
	        [&](fastgltf::sources::URI& filePath)
	        {
		        assert(filePath.fileByteOffset == 0); // we don't support offsets with stbi
		        assert(filePath.uri.isLocalPath()); // only capable of loading local files

		        const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());

		        std::filesystem::path parent = parent_path;
		        std::string current_path = parent.parent_path().string() + '/' + path; // TODO: make this cleaner

		        std::filesystem::path p = path;
		        if (p.extension() == ".ktx2")
		        {
			        new_image = basisu_load(engine, current_path.c_str());
		        }
		        else
		        {
			        unsigned char* data = stbi_load(current_path.c_str(), &width, &height, &channels, 4);
			        if (data)
			        {
				        VkExtent3D image_size{ static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
				        new_image = upload_image(engine->device, engine->graphics_queue, engine->imm_command_buffer, engine->imm_fence, engine->allocator, data, image_size, format, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, mipmapped);
				        stbi_image_free(data);
			        }
		        }
	        },
	        // TODO: handle KTX2; also verify reinterpret cast works after fastgltf 0.9 upgrade
	        [&](fastgltf::sources::Vector& vector)
	        {
		        unsigned char* data = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(vector.bytes.data()), static_cast<int>(vector.bytes.size()), &width, &height, &channels, 4);
		        if (data)
		        {
			        VkExtent3D image_size{ static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
			        new_image = upload_image(engine->device, engine->graphics_queue, engine->imm_command_buffer, engine->imm_fence, engine->allocator, data, image_size, format, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, mipmapped);

			        stbi_image_free(data);
		        }
	        },
	        // TODO: handle KTX2; also verify reinterpret cast works after fastgltf 0.9 upgrade
	        [&](fastgltf::sources::BufferView& view)
	        {
		        auto& bufferView = asset.bufferViews[view.bufferViewIndex];
		        auto& buffer = asset.buffers[bufferView.bufferIndex];
		        std::visit(fastgltf::visitor{
		                       [](auto& arg)
		                       { fmt::println("incorrect fastgltf::sources"); },
		                       [&](fastgltf::sources::Vector& vector)
		                       {
			                       unsigned char* data = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(vector.bytes.data()) + bufferView.byteOffset, static_cast<int>(bufferView.byteLength), &width, &height, &channels, 4);

			                       if (data)
			                       {
				                       VkExtent3D image_size{ static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };

				                       new_image = upload_image(engine->device, engine->graphics_queue, engine->imm_command_buffer, engine->imm_fence, engine->allocator, data, image_size, format, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, mipmapped);

				                       stbi_image_free(data);
			                       }
		                       },
		                       [&](fastgltf::sources::Array& vector)
		                       {
			                       unsigned char* data = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(vector.bytes.data()) + bufferView.byteOffset, static_cast<int>(bufferView.byteLength), &width, &height, &channels, 4);

			                       if (data)
			                       {
				                       const VkExtent3D image_size{ static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };

				                       new_image = upload_image(engine->device, engine->graphics_queue, engine->imm_command_buffer, engine->imm_fence, engine->allocator, data, image_size, format, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, mipmapped);

				                       stbi_image_free(data);
			                       }
		                       },
		                   },
		                   buffer.data);
	        }
	    },
	    image.data
	);
	// clang-format on

	// if any attempts of the above to load image data failed, we haven't written the image
	// so handle is null
	if (new_image.image == VK_NULL_HANDLE)
	{
		fmt::println("load image error");
		return {};
	}
	return new_image;
}

// TODO: refactor - try to decouple loader and engine
std::optional<std::shared_ptr<LoadedGLTF>> load_gltf(VulkanEngine* engine, Loader& loader, std::string& file_path)
{
	auto& materials_data = loader.materials;
	auto initial_materials_size = materials_data.size();
	file_path = "assets/" + file_path;
	fmt::println("Loading GLTF: {}", file_path);

	std::shared_ptr<LoadedGLTF> scene = std::make_shared<LoadedGLTF>();
	scene->creator = engine;
	LoadedGLTF& file = *scene;

	// testing extensions
	constexpr auto enabled_extensions =
	    fastgltf::Extensions::KHR_lights_punctual |
	    fastgltf::Extensions::KHR_texture_basisu;
	// fastgltf::Extensions::KHR_materials_transmission;

	fastgltf::Parser parser(enabled_extensions);
	// fastgltf::Parser parser{};

	constexpr auto gltf_options{
		fastgltf::Options::DontRequireValidAssetMember |
		// fastgltf::Options::LoadGLBBuffers | // now default behaviour
		fastgltf::Options::AllowDouble |
		fastgltf::Options::LoadExternalBuffers
	};

	std::filesystem::path path = file_path;
	auto gltf_file = fastgltf::GltfDataBuffer::FromPath(path);
	if (gltf_file.error() != fastgltf::Error::None)
		return {};

	fastgltf::Asset gltf{};

	auto type = fastgltf::determineGltfFileType(gltf_file.get());
	if (type == fastgltf::GltfType::glTF)
	{
		auto load = parser.loadGltf(gltf_file.get(), path.parent_path(), gltf_options);
		if (load)
		{
			gltf = std::move(load.get());
		}
		else
		{
			fmt::println("Failed to load gltf: {}", fastgltf::to_underlying(load.error()));
			return {};
		}
	}
	else if (type == fastgltf::GltfType::GLB)
	{
		auto load{ parser.loadGltfBinary(gltf_file.get(), path.parent_path(), gltf_options) };
		if (load)
		{
			gltf = std::move(load.get());
		}
		else
		{
			fmt::println("Failed to load gltf: {}", fastgltf::to_underlying(load.error()));
			return {};
		}
	}
	else
	{
		fmt::println("Failed to determine gltf container");
		return {};
	}

	// TODO: samplers unused, refactor
	fmt::println("gltf file has {} samplers", gltf.samplers.size());
	for (fastgltf::Sampler& sampler : gltf.samplers)
	{
		VkSamplerCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		info.maxLod = VK_LOD_CLAMP_NONE;
		info.minLod = 0;

		info.magFilter = extract_filter(sampler.magFilter.value_or(fastgltf::Filter::Nearest));
		info.minFilter = extract_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest));
		info.mipmapMode = extract_mipmap(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

		VkSampler new_sampler{};
		vkCreateSampler(engine->device, &info, nullptr, &new_sampler);

		file.samplers.push_back(new_sampler);
	}

	std::vector<std::shared_ptr<MeshAsset>> meshes{};
	std::vector<std::shared_ptr<Node>> nodes{};
	std::vector<MaterialInfo> materials{}; // id

	fmt::println("gltf file has {} images", gltf.images.size());
	std::vector<AllocatedImage> images(gltf.images.size());

	// clang-format off
	// TODO: currently supports ktx2 in URI only
	bool is_ktx2{};
	if (!gltf.images.empty()) // TODO: hack, refactor
	{
		std::visit(
		    fastgltf::visitor{
		        [](auto& arg) {},
		        [&](fastgltf::sources::URI& filePath)
		        {
			        assert(filePath.uri.isLocalPath()); // only capable of loading local files
			        const std::string filename(filePath.uri.path().begin(), filePath.uri.path().end());
			        const std::filesystem::path mypath = filename;
			        if (mypath.extension() == ".ktx2")
				        is_ktx2 = true;
		        }
		    },
		    gltf.images[0].data
		);
	}
	// clang-format on

	// if not ktx2, we defer loading during material creation so we get the right image format
	if (is_ktx2)
	{
		basist::basisu_transcoder_init();

		for (size_t idx = 0; idx < gltf.images.size(); idx++)
		{
			fastgltf::Image& image = gltf.images[idx];
			// fmt::println("image: {}", image.name.c_str()); // debug
			std::optional<AllocatedImage> img = load_image(engine, file_path, gltf, image, VK_FORMAT_R8G8B8A8_SRGB, true); // image format here is placeholder; proper format parsed from file
			if (img.has_value())
			{
				images[idx] = (*img);
				file.images[std::to_string(idx).c_str()] = images[idx];
			}
		}
	}

	fmt::println("gltf file has {} materials", gltf.materials.size());
	// const size_t materials_size = (gltf.materials.size() > 0) ? gltf.materials.size() : 1; // default material fallback

	if (gltf.materials.empty())
	{
		MaterialData mat_data{};
		// scene_material_data[0] = mat_data;
		materials_data.push_back(mat_data);
		materials.emplace_back(MaterialInfo{ MaterialPass::Opaque, 0 });
	}

	// if !is_ktx2, image loading deferred to material creation to prevent performance overhead from image_create_mutable_bit
	std::vector<bool> images_set(gltf.images.size());

	auto deferred_load = [&](size_t idx, VkFormat format)
	{
		std::optional<AllocatedImage> img{};
		if (!images_set[idx])
		{
			images_set[idx] = true;
			img = load_image(engine, file_path, gltf, gltf.images[idx], format, true);
			if (img.has_value())
			{
				images[idx] = (*img);
				file.images[std::to_string(idx).c_str()] = images[idx];
			}
			else
			{
				images[idx] = engine->error_image;
			}
		}
	};

	bool using_basisu = false;
	for (auto& e : gltf.extensionsRequired)
	{
		if (e == "KHR_texture_basisu")
			using_basisu = true;
	}

	// TODO: need to implement MaterialCache as its common for gltf to have same material under different name
	// current implementation simply duplicates this in the material buffer
	int material_idx{ 0 };
	for (fastgltf::Material& mat : gltf.materials)
	{
		// fmt::println("material: {}", mat.name.c_str()); // debug

		MaterialData mat_data{};
		mat_data.base_color_factor.x = mat.pbrData.baseColorFactor[0];
		mat_data.base_color_factor.y = mat.pbrData.baseColorFactor[1];
		mat_data.base_color_factor.z = mat.pbrData.baseColorFactor[2];
		mat_data.base_color_factor.w = mat.pbrData.baseColorFactor[3];
		mat_data.metallic_factor = mat.pbrData.metallicFactor;
		mat_data.roughness_factor = mat.pbrData.roughnessFactor;

		if (mat.pbrData.baseColorTexture.has_value())
		{
			size_t idx = is_ktx2 && using_basisu
			                 ? gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].basisuImageIndex.value()
			                 : gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
			// size_t sampler = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].samplerIndex.value();

			if (!is_ktx2)
				deferred_load(idx, VK_FORMAT_R8G8B8A8_SRGB);
			AllocatedImage img = images[idx];

			mat_data.diffuse_id = engine->texture_cache.add_texture(img.view); // img guaranteed to have value
		}

		if (mat.pbrData.metallicRoughnessTexture.has_value())
		{
			size_t idx = is_ktx2 && using_basisu
			                 ? gltf.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].basisuImageIndex.value()
			                 : gltf.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].imageIndex.value();
			// size_t sampler{ gltf.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].samplerIndex.value() };

			if (!is_ktx2)
				deferred_load(idx, VK_FORMAT_R8G8B8A8_UNORM);
			AllocatedImage img = images[idx];

			mat_data.metal_roughness_id = engine->texture_cache.add_texture(img.view);
		}

		if (mat.normalTexture.has_value())
		{
			size_t idx = is_ktx2 && using_basisu
			                 ? gltf.textures[mat.normalTexture.value().textureIndex].basisuImageIndex.value()
			                 : gltf.textures[mat.normalTexture.value().textureIndex].imageIndex.value();

			if (!is_ktx2)
				deferred_load(idx, VK_FORMAT_R8G8B8A8_UNORM);
			AllocatedImage img = images[idx];

			mat_data.normal_id = engine->texture_cache.add_texture(img.view);
		}

		if (mat.occlusionTexture.has_value())
		{
			size_t idx = is_ktx2 && using_basisu
			                 ? gltf.textures[mat.occlusionTexture.value().textureIndex].basisuImageIndex.value()
			                 : gltf.textures[mat.occlusionTexture.value().textureIndex].imageIndex.value();

			if (!is_ktx2)
				deferred_load(idx, VK_FORMAT_R8G8B8A8_UNORM);
			AllocatedImage img = images[idx];

			mat_data.occlusion_id = engine->texture_cache.add_texture(img.view);
		}

		if (mat.emissiveTexture.has_value())
		{
			size_t idx = is_ktx2 && using_basisu
			                 ? gltf.textures[mat.emissiveTexture.value().textureIndex].basisuImageIndex.value()
			                 : gltf.textures[mat.emissiveTexture.value().textureIndex].imageIndex.value();

			if (!is_ktx2)
				deferred_load(idx, VK_FORMAT_R8G8B8A8_SRGB);
			AllocatedImage img = images[idx];

			mat_data.emissive_id = engine->texture_cache.add_texture(img.view);
		}

		materials_data.push_back(mat_data);

		MaterialPass pass_type = MaterialPass::Opaque;
		switch (mat.alphaMode)
		{
		case fastgltf::AlphaMode::Mask:
			pass_type = MaterialPass::Mask;
			break;
		case fastgltf::AlphaMode::Blend:
			pass_type = MaterialPass::Blend;
			break;
		default:
			break;
		}

		uint32_t double_sided = mat.doubleSided;

		materials.emplace_back(MaterialInfo{ pass_type, double_sided });
		material_idx++;
	}

	auto& combined_indices = loader.combined_indices;
	auto& combined_vertices = loader.combined_vertices;
	auto& meshlet_indices = loader.meshlet_indices;
	auto& meshlets = loader.meshlets;

	fmt::println("gltf file has {} meshes", gltf.meshes.size());

	auto mesh_idx = 0;
	for (fastgltf::Mesh& mesh : gltf.meshes)
	{
		std::shared_ptr<MeshAsset> new_mesh{ std::make_shared<MeshAsset>() };
		meshes.push_back(new_mesh);
		file.meshes[std::to_string(mesh_idx).c_str()] = new_mesh;
		mesh_idx++;
		new_mesh->name = mesh.name;

		for (auto&& p : mesh.primitives)
		{
			std::vector<Vertex> vertices{};
			std::vector<uint32_t> indices{};

			GeoSurface new_surface{};

			// clang-format off
			{
				fastgltf::Accessor& index_accessor = gltf.accessors[p.indicesAccessor.value()];
				indices.reserve(index_accessor.count);

				fastgltf::iterateAccessor<std::uint32_t>(gltf, index_accessor, [&](std::uint32_t idx) { indices.push_back(idx);});
			}

			// load vertex positions
			{
				fastgltf::Accessor& pos_accessor = gltf.accessors[p.findAttribute("POSITION")->accessorIndex];
				vertices.resize(pos_accessor.count);

				fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, pos_accessor, [&](glm::vec3 v, size_t index)
				    {
						Vertex new_vtx{};
						new_vtx.position = v;
						vertices[index] = new_vtx;
				    }
				);
			}

			// load vertex normals
			{
				auto normals = p.findAttribute("NORMAL");
				if (normals != p.attributes.end())
				{
					fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[normals->accessorIndex], [&](glm::vec3 v, size_t index){ vertices[index].normal = v; });
				}
			}

			bool generate_tangents{};
			// load vertex tangents
			{
				auto tangents = p.findAttribute("TANGENT");
				if (tangents != p.attributes.end())
				{
					fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[tangents->accessorIndex], [&](glm::vec4 v, size_t index){ vertices[index].tangent = v; });
				}
				else
				{
					generate_tangents = true;
				}
			}

			// mikk tangent generation
			if (generate_tangents)
			{
				fmt::println("generating tangents manually");
				MikkMesh mikk_mesh{ &vertices, &indices };
				calculateTangents(mikk_mesh);
			}

			// load uvs
			{
				auto uv = p.findAttribute("TEXCOORD_0");
				if (uv != p.attributes.end())
				{
					fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[uv->accessorIndex], [&](glm::vec2 v, size_t index)
					    {
							vertices[index].uv_x = v.x;
							vertices[index].uv_y = v.y;
					    }
					);
				}
			}
			// clang-format on

			new_surface.vertex_offset = static_cast<uint32_t>(combined_vertices.size());

			// meshoptimizer step
			optimize_mesh(vertices, indices, meshlet_indices, meshlets, new_surface, combined_vertices, combined_indices);
			combined_vertices.insert(combined_vertices.end(), vertices.begin(), vertices.end());

			if (p.materialIndex.has_value())
			{
				size_t idx = p.materialIndex.value();
				auto [pass_type, double_sided] = materials[idx];
				new_surface.material_id = static_cast<uint32_t>(idx + initial_materials_size);
				new_surface.pass = pass_type;
			}
			else
			{
				// TODO: refactor - mesh has no material, assign first material
				auto [pass_type, double_sided] = materials[0];
				new_surface.material_id = 0;
				new_surface.pass = pass_type;
			}

			new_mesh->surfaces.push_back(new_surface);
		}
	}

	// load all nodes and their meshes
	auto node_idx = 0;
	for (fastgltf::Node& node : gltf.nodes)
	{
		std::shared_ptr<Node> new_node{};

		if (node.meshIndex.has_value())
		{
			new_node = std::make_shared<Node>();
			new_node->mesh = meshes[*(node.meshIndex)];
		}
		else
		{
			fmt::println("node has no mesh: ", node.name.c_str());
			new_node = std::make_shared<Node>(); // TODO: refactor? absorbing allocation cost for dummy node
		}

		nodes.push_back(new_node);
		file.nodes[std::to_string(node_idx).c_str()] = new_node;
		node_idx++;

		// clang-format off
		std::visit(fastgltf::visitor{
				[&](fastgltf::math::fmat4x4 matrix)
		        {
		            memcpy(&new_node->local_transform, matrix.data(), sizeof(matrix));
		        },
		        [&](fastgltf::TRS transform)
		        {
		            const glm::vec3 tl(transform.translation[0], transform.translation[1], transform.translation[2]);
		            const glm::quat rot(transform.rotation[3], transform.rotation[0], transform.rotation[1], transform.rotation[2]);
		            const glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);

		            const glm::mat4 tm = glm::translate(glm::mat4(1.f), tl);
		            const glm::mat4 rm = glm::toMat4(rot);
		            const glm::mat4 sm = glm::scale(glm::mat4(1.f), sc);

		            new_node->local_transform = tm * rm * sm;
		        }
			},
		    node.transform
		);
		// clang-format on
	}

	for (int i = 0; i < gltf.nodes.size(); i++)
	{
		fastgltf::Node& node = gltf.nodes[i];
		std::shared_ptr<Node>& scene_node = nodes[i];

		for (auto& c : node.children)
		{
			scene_node->children.push_back(nodes[c]);
			nodes[c]->parent = scene_node;
		}
	}

	for (auto& node : nodes)
	{
		if (node->parent.lock() == nullptr)
		{
			file.top_nodes.push_back(node);
			node->refresh_transform(glm::mat4(1.0f));
		}
	}

	fmt::println("size of topnodes: {}", scene->top_nodes.size());
	fmt::println("size of nodes: {}", scene->nodes.size());
	fmt::println("size of gltf nodes: {}", gltf.nodes.size());

	return scene;
}

void LoadedGLTF::clear()
{
	const VkDevice device = creator->device;
	const VmaAllocator allocator = creator->allocator;

	for (auto& img : images | std::views::values)
	{
		destroy_image(device, allocator, img);
	}

	for (const auto s : samplers)
	{
		vkDestroySampler(device, s, nullptr);
	}
}

void calculateTangents(MikkMesh& m)
{
	SMikkTSpaceInterface mikkInterface{};
	mikkInterface.m_getNumFaces = mikk_getNumFaces;
	mikkInterface.m_getNumVerticesOfFace = mikk_getNumVerticesOfFace;
	mikkInterface.m_getPosition = mikk_getPosition;
	mikkInterface.m_getNormal = mikk_getNormal;
	mikkInterface.m_getTexCoord = mikk_getTexCoord;
	mikkInterface.m_setTSpaceBasic = mikk_setTSpaceBasic;
	mikkInterface.m_setTSpace = nullptr;

	SMikkTSpaceContext mikkContext{};
	mikkContext.m_pInterface = &mikkInterface;
	mikkContext.m_pUserData = &m;

	genTangSpaceDefault(&mikkContext);
}

int mikk_getNumFaces(const SMikkTSpaceContext* context)
{
	MikkMesh mesh = *(static_cast<MikkMesh*>(context->m_pUserData));
	return static_cast<int>((mesh.indices)->size()) / 3;
}

int mikk_getNumVerticesOfFace(const SMikkTSpaceContext* context, int faceIndex)
{
	return 3;
}

void mikk_getPosition(const SMikkTSpaceContext* context, float outPosition[3], int faceIndex, int vertIndex)
{
	MikkMesh mesh = *(static_cast<MikkMesh*>(context->m_pUserData));
	uint32_t idx = (*mesh.indices)[faceIndex * 3 + vertIndex];
	glm::vec3 pos = (*mesh.vertices)[idx].position;
	outPosition[0] = pos.x;
	outPosition[1] = pos.y;
	outPosition[2] = pos.z;
}

void mikk_getNormal(const SMikkTSpaceContext* context, float outNormal[3], int faceIndex, int vertIndex)
{
	MikkMesh mesh = *(static_cast<MikkMesh*>(context->m_pUserData));
	uint32_t idx = (*mesh.indices)[faceIndex * 3 + vertIndex];
	glm::vec3 normal = (*mesh.vertices)[idx].normal;
	outNormal[0] = normal.x;
	outNormal[1] = normal.y;
	outNormal[2] = normal.z;
}

void mikk_getTexCoord(const SMikkTSpaceContext* context, float outUV[2], int faceIndex, int vertIndex)
{
	MikkMesh mesh = *(static_cast<MikkMesh*>(context->m_pUserData));
	uint32_t idx = (*mesh.indices)[faceIndex * 3 + vertIndex];
	glm::vec2 uv = glm::vec2((*mesh.vertices)[idx].uv_x, (*mesh.vertices)[idx].uv_y);
	outUV[0] = uv.x;
	outUV[1] = uv.y;
}

void mikk_setTSpaceBasic(const SMikkTSpaceContext* context, const float outTangent[3], float sign, int faceIndex, int vertIndex)
{
	MikkMesh mesh = *(static_cast<MikkMesh*>(context->m_pUserData));
	uint32_t idx = (*mesh.indices)[faceIndex * 3 + vertIndex];
	glm::vec4& tangent = (*mesh.vertices)[idx].tangent;
	tangent.x = outTangent[0];
	tangent.y = outTangent[1];
	tangent.z = outTangent[2];
	tangent.w = -sign;
}