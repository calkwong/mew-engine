#include "common.h"
#include "config.h"
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

// update
#include <ranges>
#include <algorithm>
#include <execution>
#include <cassert>
#include <cstddef>

namespace
{
	struct MikkMesh
	{
		std::vector<Vertex>* vertices{};
		std::vector<uint32_t>* indices{};
	};

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

	std::vector<AllocatedImage> load_images(const fastgltf::Asset& asset, VulkanEngine* engine, std::string_view asset_path)
	{
		std::vector<AllocatedImage> images{};

		struct RawImageData
		{
			int width{};
			int height{};
			int components{};
			uint32_t mips{};
			uint32_t size{};
			bool is_ktx2{};
			VkFormat format{};

			std::unique_ptr<unsigned char[], decltype([](unsigned char* p){ stbi_image_free(p); })> data{};
			std::unique_ptr<unsigned char[]> ktx{};
			std::unique_ptr<basist::ktx2_image_level_info[]> ktx_info{};
		};

		// TODO: use fastgltf::mimetype for robustness
		auto create_raw_image_data = [&](const std::filesystem::path& full_path, bool is_ktx2) -> RawImageData
		{
			RawImageData raw_image_data{};

			raw_image_data.is_ktx2 = is_ktx2;

			if (is_ktx2)
			{
				std::vector<uint8_t> buffer{};

				if (!read_ktx2_file(full_path.string().c_str(), buffer))
					assert(0);

				// create the KTX2 transcoder object
				basist::ktx2_transcoder transcoder{};

				// initialize the transcoder
				if (!transcoder.init(buffer.data(), static_cast<uint32_t>(buffer.size())))
					assert(0);

				// TODO: refactor when we stop using BC7 across the board
				auto target_format = basist::transcoder_texture_format::cTFBC7_RGBA;
				uint32_t bytes_per_block_or_pixel = basist::basis_get_bytes_per_block_or_pixel(target_format);

				auto transfer_func = transcoder.get_dfd_transfer_func();
				switch (transfer_func)
				{
				case basist::KTX2_KHR_DF_TRANSFER_SRGB:
					raw_image_data.format = VK_FORMAT_BC7_SRGB_BLOCK;
					break;
				case basist::KTX2_KHR_DF_TRANSFER_LINEAR:
					raw_image_data.format = VK_FORMAT_BC7_UNORM_BLOCK;
					break;
				default:
					assert(0);
				}

				raw_image_data.mips = transcoder.get_levels();
				raw_image_data.ktx_info = std::make_unique<basist::ktx2_image_level_info[]>(raw_image_data.mips);

				for (uint32_t i = 0; i < raw_image_data.mips; i++)
				{
					transcoder.get_image_level_info(raw_image_data.ktx_info[i], i, 0, 0);
				}

				uint32_t num_blocks_or_pixels{};
				uint64_t buffer_size{};
				for (uint32_t i = 0; i < raw_image_data.mips; i++)
				{
					num_blocks_or_pixels = raw_image_data.ktx_info[i].m_total_blocks;
					buffer_size += bytes_per_block_or_pixel * num_blocks_or_pixels;
				}

				auto header = transcoder.get_header();
				auto supercompression_scheme = header.m_supercompression_scheme;
				if (supercompression_scheme == basist::KTX2_SS_NONE)
					assert(0); // TODO: we assume texture is always supercompressed

				transcoder.start_transcoding();

				raw_image_data.ktx = std::make_unique<unsigned char[]>(buffer_size);

				unsigned char* ktx_data = raw_image_data.ktx.get();

				raw_image_data.size = 0;
				for (uint32_t i = 0; i < raw_image_data.mips; i++)
				{
					num_blocks_or_pixels = raw_image_data.ktx_info[i].m_total_blocks;
					uint32_t output_size = bytes_per_block_or_pixel * num_blocks_or_pixels;
					if (!transcoder.transcode_image_level(i, 0, 0, ktx_data, output_size, target_format))
						assert(0); // TODO: ?
					ktx_data += output_size;
					raw_image_data.size += output_size;
				}
			}
			else
			{
				raw_image_data.data.reset(stbi_load(full_path.string().c_str(), &raw_image_data.width, &raw_image_data.height, &raw_image_data.components, 4));
				// raw_image_data.mips = static_cast<uint32_t>(std::floor(std::log2(std::max(raw_image_data.width, raw_image_data.height)))) + 1;
				raw_image_data.mips = 1;
				raw_image_data.format = VK_FORMAT_R8G8B8A8_UNORM;
				raw_image_data.size = static_cast<uint32_t>(raw_image_data.width * raw_image_data.height * 4);
			}

			return raw_image_data;
		};

		// generates sequence of value by repeatedly incrementing initial value up to bound - types must match!
		const auto indices = std::ranges::iota_view(static_cast<size_t>(0), asset.images.size());

		auto has_ktx2_format = [&](std::string_view file_path) -> bool
		{
			size_t pos = file_path.find(".ktx2");
			return (pos != std::string::npos);
		};

		auto raw_images = std::vector<RawImageData>(asset.images.size());

		std::filesystem::path current_path = asset_path;
		basist::basisu_transcoder_init();

		// TODO: potentially problematic in parallel
		std::transform(std::execution::par, indices.begin(), indices.end(), raw_images.begin(), [&](size_t index)
			{
				const fastgltf::Image& image = asset.images[index];

				// how does this fare against std::variant?
				if (const auto* file_path = std::get_if<fastgltf::sources::URI>(&image.data))
				{
					assert(file_path->fileByteOffset == 0); // we don't support offsets with stbi
					assert(file_path->uri.isLocalPath()); // only load local files

					bool is_ktx2 = has_ktx2_format(file_path->uri.path());
					std::filesystem::path full_path = current_path / file_path->uri.path();

					return create_raw_image_data(full_path, is_ktx2);
				}
				// TODO
				if (const auto* file_path = std::get_if<fastgltf::sources::Vector>(&image.data))
				{
					assert(0 && "fastgltf::sources::Vector not implemented");
					return RawImageData{};
				}
				// TODO
				if (const auto* file_path = std::get_if<fastgltf::sources::BufferView>(&image.data))
				{
					assert(0 && "fastgltf::sources::BufferView not implemented");
					return RawImageData{};
				}

				assert(0);
				return RawImageData{};

			}
		);

		// create staging buffer
		AllocatedBuffer scratch = create_buffer(engine->allocator, 1000000000, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

		struct ImageUploadInfo
		{
			const void* data{};
			uint32_t size{};
			uint32_t buffer_offset{};
			uint32_t image_index{};
			uint32_t mips{};
			VkExtent3D extent{};
		};

		// TODO: refactor when we stop using BC7 across the board
		auto target_format = basist::transcoder_texture_format::cTFBC7_RGBA;
		const uint32_t bytes_per_block_or_pixel = basist::basis_get_bytes_per_block_or_pixel(target_format);

		// create image upload info
		std::vector<ImageUploadInfo> image_upload_info{};
		uint32_t buffer_offset{}; // offset for each image upload
		for (const auto& raw_image_data : raw_images)
		{
			if (raw_image_data.is_ktx2)
			{
				uint32_t upload_offset{};
				for (uint32_t mip = 0; mip < raw_image_data.mips; mip++)
				{
					uint32_t num_blocks_or_pixels = raw_image_data.ktx_info[mip].m_total_blocks;
					uint32_t output_size = bytes_per_block_or_pixel * num_blocks_or_pixels;

					image_upload_info.emplace_back(ImageUploadInfo{
						.data = static_cast<void*>(raw_image_data.ktx.get() + upload_offset),
						.size = output_size,
						.buffer_offset = buffer_offset,
						.image_index = static_cast<uint32_t>(images.size()),
						.mips = mip,
						.extent = { raw_image_data.ktx_info[mip].m_orig_width, raw_image_data.ktx_info[mip].m_orig_width, 1 }
					});

					upload_offset += output_size;
					buffer_offset += output_size;
				}

				images.emplace_back(create_image(engine->device, engine->allocator, { raw_image_data.ktx_info[0].m_orig_width, raw_image_data.ktx_info[0].m_orig_height, 1}, raw_image_data.format,
					VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, true
				));
			}
			else
			{
				image_upload_info.emplace_back(ImageUploadInfo{
					.data = static_cast<void*>(raw_image_data.data.get()),
					.size = raw_image_data.size,
					.buffer_offset = buffer_offset,
					.image_index = static_cast<uint32_t>(images.size()),
					.mips = 0,
					.extent = VkExtent3D{ static_cast<uint32_t>(raw_image_data.width), static_cast<uint32_t>(raw_image_data.height), 1 }
				});

				buffer_offset += raw_image_data.size;

				images.emplace_back(create_image(engine->device, engine->allocator, { static_cast<uint32_t>(raw_image_data.width), static_cast<uint32_t>(raw_image_data.height), 1 },
					VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, false
				));
			}
		}

		for (const auto& image : images)
		{
			engine->texture_cache.add_texture(image.view);
		}

		// note: flush image uploads - call this in immediate submit
		auto flush_uploads = [&]()
		{
			// note: buffer_offset already accounts for entire data size?
			if (scratch.info.size < buffer_offset + image_upload_info.back().size)
			{
				destroy_buffer(engine->allocator, scratch);
				scratch = create_buffer(engine->allocator, buffer_offset * 1.5, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
			}

			// copy to staging in parallel
			std::for_each(std::execution::par, image_upload_info.begin(), image_upload_info.end(), [&](const ImageUploadInfo& upload_info)
				{
					auto p = static_cast<std::byte*>(scratch.info.pMappedData) + upload_info.buffer_offset; // this needs an overall offset
					memcpy(p, upload_info.data, upload_info.size); // this needs a local offset
				}
			);

			std::vector<VkBufferImageCopy2> buffer_image_copies(image_upload_info.size());
			std::vector<VkCopyBufferToImageInfo2> buffer_to_image_info(image_upload_info.size());
			for (int i = 0; i < image_upload_info.size(); i++)
			{
				const auto& upload_info = image_upload_info[i];
				auto& copy = buffer_image_copies[i];
				auto& info = buffer_to_image_info[i];

				copy.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
				copy.pNext = nullptr;
				copy.bufferOffset = upload_info.buffer_offset;
				copy.bufferRowLength = 0;
				copy.bufferImageHeight = 0;
				copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				copy.imageSubresource.mipLevel = upload_info.mips;
				copy.imageSubresource.baseArrayLayer = 0;
				copy.imageSubresource.layerCount = 1;
				copy.imageOffset = VkOffset3D{ 0,0,0 };
				copy.imageExtent = upload_info.extent;

				info.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
				info.pNext = nullptr;
				info.srcBuffer = scratch.buffer;
				info.dstImage = images[upload_info.image_index].image;
				info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				info.regionCount = 1;
				info.pRegions = &copy;
			}

			for (int i = 0; i < image_upload_info.size(); i++)
			{
				// TODO: handle mipmaps for non-ktx path
				vkCmdCopyBufferToImage2(engine->imm_command_buffer, &buffer_to_image_info[i]);
			}
		};

		// build list of barriers
		std::vector<VkImageMemoryBarrier2> image_barriers(asset.images.size());
		for (int i = 0; i < image_barriers.size(); i++)
		{
			image_barriers[i] = image_barrier(images[i].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0, VK_ACCESS_2_TRANSFER_WRITE_BIT
			);
		}

		immediate_submit(engine->device, engine->graphics_queue, engine->imm_command_buffer, engine->imm_fence, [&](VkCommandBuffer cmd)
			{
				pipeline_barrier(engine->imm_command_buffer, nullptr, 0, image_barriers.data(), image_barriers.size());

				flush_uploads();

				for (int i = 0; i < image_barriers.size(); i++)
				{
					image_barriers[i] = image_barrier(images[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
						VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
					);
				}

				pipeline_barrier(engine->imm_command_buffer, nullptr, 0, image_barriers.data(), image_barriers.size());
			}
		);

		destroy_buffer(engine->allocator, scratch);

		std::for_each(std::execution::par, raw_images.begin(), raw_images.end(), [&](RawImageData& raw_image_data)
		{
			raw_image_data.data.reset();
			raw_image_data.ktx.reset();
			raw_image_data.ktx_info.reset();
		});

		return images;
	}

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

			float px = meshopt_dequantizeHalf(vertices[i].px);
			float py = meshopt_dequantizeHalf(vertices[i].py);
			float pz = meshopt_dequantizeHalf(vertices[i].pz);

			positions[i] = glm::vec3(px, py, pz);
			center += positions[i];
		}

		std::vector<glm::vec3> normals(vertex_count);
		for (size_t i = 0; i < vertex_count; i++)
		{
			auto n = vertices[i].normal;
			glm::vec3 normal = glm::vec3((n >> 20) & 1023, (n >> 10) & 1023, n & 1023) / glm::vec3(511.0) - glm::vec3(1.0);
			normals[i] = normal;
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
		constexpr size_t max_vertices = MESHLET_MAX_VERTICES;
		constexpr size_t max_triangles = MESHLET_MAX_TRIANGLES;
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

				new_meshlet.cx = meshopt_quantizeHalf(bounds.center[0]);
				new_meshlet.cy = meshopt_quantizeHalf(bounds.center[1]);
				new_meshlet.cz = meshopt_quantizeHalf(bounds.center[2]);

				new_meshlet.radius = meshopt_quantizeHalf(bounds.radius);

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

		float px = meshopt_dequantizeHalf((*mesh.vertices)[idx].px);
		float py = meshopt_dequantizeHalf((*mesh.vertices)[idx].py);
		float pz = meshopt_dequantizeHalf((*mesh.vertices)[idx].pz);

		outPosition[0] = px;
		outPosition[1] = py;
		outPosition[2] = pz;
	}

	void mikk_getNormal(const SMikkTSpaceContext* context, float outNormal[3], int faceIndex, int vertIndex)
	{
		MikkMesh mesh = *(static_cast<MikkMesh*>(context->m_pUserData));
		uint32_t idx = (*mesh.indices)[faceIndex * 3 + vertIndex];

		uint32_t n = (*mesh.vertices)[idx].normal;
		glm::vec3 normal = normalize(glm::vec3((n >> 20) & 1023, (n >> 10) & 1023, n & 1023) / glm::vec3(511.0) - glm::vec3(1.0)); // normalize or no?

		outNormal[0] = normal.x;
		outNormal[1] = normal.y;
		outNormal[2] = normal.z;
	}

	void mikk_encodeOct(float& x, float& y, float z)
	{
		float sum = abs(x) + abs(y) + abs(z);
		x /= sum;
		y /= sum;

		// sign doesnt change so we can omit this
		// z /= sum;

		float u = z >= 0.0f ? x : (1.0f - abs(y)) * (x >= 0.0f ? 1.0f : -1.0f);
		float v = z >= 0.0f ? y : (1.0f - abs(x)) * (y >= 0.0f ? 1.0f : -1.0f);

		// optional mapping to [0,1]?

		x = u;
		y = v;
	}

	void mikk_getTexCoord(const SMikkTSpaceContext* context, float outUV[2], int faceIndex, int vertIndex)
	{
		MikkMesh mesh = *(static_cast<MikkMesh*>(context->m_pUserData));
		uint32_t idx = (*mesh.indices)[faceIndex * 3 + vertIndex];

		float uvx = meshopt_dequantizeHalf((*mesh.vertices)[idx].uv_x);
		float uvy = meshopt_dequantizeHalf((*mesh.vertices)[idx].uv_y);

		outUV[0] = uvx;
		outUV[1] = uvy;
	}

	void mikk_setTSpaceBasic(const SMikkTSpaceContext* context, const float outTangent[3], float sign, int faceIndex, int vertIndex)
	{
		MikkMesh mesh = *(static_cast<MikkMesh*>(context->m_pUserData));
		uint32_t idx = (*mesh.indices)[faceIndex * 3 + vertIndex];

		// tangent.w = -sign;

		auto tx = outTangent[0];
		auto ty = outTangent[1];
		auto tz = outTangent[2];
		mikk_encodeOct(tx, ty, tz);

		uint16_t t =
			(meshopt_quantizeSnorm(tx, 8) + 127) << 8 |
			(meshopt_quantizeSnorm(ty, 8) + 127);

		(*mesh.vertices)[idx].tangent = t;
		(*mesh.vertices)[idx].normal |= (-sign >= 0 ? 1 : 0) << 30;
	}

	void mikk_calculate_tangents(MikkMesh& m)
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

	constexpr auto supported_extensions =
		fastgltf::Extensions::KHR_lights_punctual |
		fastgltf::Extensions::KHR_texture_basisu;
	// fastgltf::Extensions::KHR_materials_transmission;

	fastgltf::Parser parser(supported_extensions);

	// TODO: look up options
	constexpr auto gltf_options{
		fastgltf::Options::DontRequireValidAssetMember |
		// fastgltf::Options::LoadGLBBuffers | // now default behaviour
		fastgltf::Options::AllowDouble |
		fastgltf::Options::LoadExternalBuffers
	};

	std::filesystem::path path = file_path;
	file.asset_path = path.parent_path().string();
	auto gltf_file = fastgltf::GltfDataBuffer::FromPath(path);

	if (gltf_file.error() != fastgltf::Error::None)
		return {};

	fastgltf::Asset asset{};

	auto type = fastgltf::determineGltfFileType(gltf_file.get());
	if (type == fastgltf::GltfType::glTF)
	{
		auto load = parser.loadGltf(gltf_file.get(), path.parent_path(), gltf_options);
		if (load)
		{
			asset = std::move(load.get());
		}
		else
		{
			fmt::println("Failed to load gltf: {}", fastgltf::to_underlying(load.error()));
			return {};
		}
	}
	else if (type == fastgltf::GltfType::GLB)
	{
		auto load{parser.loadGltfBinary(gltf_file.get(), path.parent_path(), gltf_options)};
		if (load)
		{
			asset = std::move(load.get());
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

	std::vector<std::shared_ptr<MeshAsset>> meshes{};
	std::vector<std::shared_ptr<Node>> nodes{};

	// note: handle another way
	assert(!asset.materials.empty());

	size_t texture_cache_offset = engine->texture_cache.image_infos.size(); // important! do this before loading images

	// TODO: currently supports ktx2 in URI only
	std::vector<AllocatedImage> images{};
	auto start = std::chrono::system_clock::now();
	if (!asset.images.empty())
		images = load_images(asset, engine, file.asset_path);
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	float ret = static_cast<float>(elapsed.count()) / 1000.0f;
	fmt::println("load_images: {}ms", ret);

	// TODO: do we really need an unordered map here?
	for (int i = 0; i < images.size(); i++)
	{
		file.images[std::to_string(i).c_str()] = images[i];
	}

	for (fastgltf::Material& mat : asset.materials)
	{
		MaterialData mat_data{};
		mat_data.base_color_factor.x = mat.pbrData.baseColorFactor[0];
		mat_data.base_color_factor.y = mat.pbrData.baseColorFactor[1];
		mat_data.base_color_factor.z = mat.pbrData.baseColorFactor[2];
		mat_data.base_color_factor.w = mat.pbrData.baseColorFactor[3];
		mat_data.metallic_factor = mat.pbrData.metallicFactor;
		mat_data.roughness_factor = mat.pbrData.roughnessFactor;

		if (mat.pbrData.baseColorTexture.has_value())
		{
			size_t image_index = asset.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex
				                     ? asset.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.
				                     value()
				                     : asset.textures[mat.pbrData.baseColorTexture.value().textureIndex].
				                       basisuImageIndex.value();

			mat_data.diffuse_id = static_cast<uint32_t>(texture_cache_offset + image_index);
		}

		if (mat.pbrData.metallicRoughnessTexture.has_value())
		{
			size_t image_index = asset.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].imageIndex
				                     ? asset.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].
				                       imageIndex.value()
				                     : asset.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].
				                       basisuImageIndex.value();

			mat_data.metal_roughness_id = static_cast<uint32_t>(texture_cache_offset + image_index);
		}

		if (mat.normalTexture.has_value())
		{
			size_t image_index = asset.textures[mat.normalTexture.value().textureIndex].imageIndex
				                     ? asset.textures[mat.normalTexture.value().textureIndex].imageIndex.value()
				                     : asset.textures[mat.normalTexture.value().textureIndex].basisuImageIndex.value();

			mat_data.normal_id = static_cast<uint32_t>(texture_cache_offset + image_index);
		}

		if (mat.occlusionTexture.has_value())
		{
			size_t image_index = asset.textures[mat.occlusionTexture.value().textureIndex].imageIndex
				                     ? asset.textures[mat.occlusionTexture.value().textureIndex].imageIndex.value()
				                     : asset.textures[mat.occlusionTexture.value().textureIndex].basisuImageIndex.
				                     value();

			mat_data.occlusion_id = static_cast<uint32_t>(texture_cache_offset + image_index);
		}

		if (mat.emissiveTexture.has_value())
		{
			size_t image_index = asset.textures[mat.emissiveTexture.value().textureIndex].imageIndex
				                     ? asset.textures[mat.emissiveTexture.value().textureIndex].imageIndex.value()
				                     : asset.textures[mat.emissiveTexture.value().textureIndex].basisuImageIndex.
				                     value();

			mat_data.emissive_id = static_cast<uint32_t>(texture_cache_offset + image_index);
		}

		materials_data.push_back(mat_data);
	}

	auto& combined_indices = loader.combined_indices;
	auto& combined_vertices = loader.combined_vertices;
	auto& meshlet_indices = loader.meshlet_indices;
	auto& meshlets = loader.meshlets;

	auto mesh_idx = 0;
	for (fastgltf::Mesh& mesh : asset.meshes)
	{
		std::shared_ptr<MeshAsset> new_mesh{std::make_shared<MeshAsset>()};
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
				fastgltf::Accessor& index_accessor = asset.accessors[p.indicesAccessor.value()];
				indices.reserve(index_accessor.count);

				fastgltf::iterateAccessor<std::uint32_t>(asset, index_accessor, [&](std::uint32_t idx) { indices.push_back(idx);});
			}

			// load vertex positions
			{
				fastgltf::Accessor& pos_accessor = asset.accessors[p.findAttribute("POSITION")->accessorIndex];
				vertices.resize(pos_accessor.count);

				fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, pos_accessor, [&](glm::vec3 v, size_t index)
				    {
						Vertex new_vtx{};

						new_vtx.px = meshopt_quantizeHalf(v.x);
						new_vtx.py = meshopt_quantizeHalf(v.y);
						new_vtx.pz = meshopt_quantizeHalf(v.z);

						vertices[index] = new_vtx;
				    }
				);
			}

			// load vertex normals
			{
				auto normals = p.findAttribute("NORMAL");
				if (normals != p.attributes.end())
				{
					fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, asset.accessors[normals->accessorIndex], [&](glm::vec3 v, size_t index)
						{
							uint32_t normal =
								(meshopt_quantizeSnorm(v.x, 10) + 511) << 20 |
								(meshopt_quantizeSnorm(v.y, 10) + 511) << 10 |
								(meshopt_quantizeSnorm(v.z, 10) + 511);

							vertices[index].normal = normal;
						}
					);
				}
			}

			auto encode_oct = [&](glm::vec3 n) -> glm::vec2 {
				n /= (abs(n.x) + abs(n.y) + abs(n.z));
				float u = n.z >= 0.0f ? n.x : (1.0f - abs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f);
				float v = n.z >= 0.0f ? n.y : (1.0f - abs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f);

				// optional mapping to [0, 1]?
				return glm::vec2(u, v);
			};

			// load uvs
			{
				auto uv = p.findAttribute("TEXCOORD_0");
				if (uv != p.attributes.end())
				{
					fastgltf::iterateAccessorWithIndex<glm::vec2>(asset, asset.accessors[uv->accessorIndex], [&](glm::vec2 v, size_t index)
						{
						  vertices[index].uv_x = meshopt_quantizeHalf(v.x);
						  vertices[index].uv_y = meshopt_quantizeHalf(v.y);
						}
					);
				}
			}

			bool generate_tangents{};
			// load vertex tangents
			{
				auto tangents = p.findAttribute("TANGENT");
				if (tangents != p.attributes.end())
				{
					fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, asset.accessors[tangents->accessorIndex], [&](glm::vec4 v, size_t index)
						{
							glm::vec2 t = encode_oct(glm::vec3(v));

							uint16_t tangent =
								(meshopt_quantizeSnorm(t.x, 8) + 127) << 8 |
								(meshopt_quantizeSnorm(t.y, 8) + 127);

							vertices[index].tangent = tangent;
							vertices[index].normal |= (v.w >= 0 ? 1 : 0) << 30;
						}
					);
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
				mikk_calculate_tangents(mikk_mesh);
			}

			// clang-format on

			new_surface.vertex_offset = static_cast<uint32_t>(combined_vertices.size());

			// meshoptimizer step
			optimize_mesh(vertices, indices, meshlet_indices, meshlets, new_surface, combined_vertices,
			              combined_indices);
			combined_vertices.insert(combined_vertices.end(), vertices.begin(), vertices.end());

			if (p.materialIndex.has_value())
			{
				size_t idx = p.materialIndex.value();
				auto alpha_mode = asset.materials[idx].alphaMode;
				switch (alpha_mode)
				{
				case fastgltf::AlphaMode::Mask:
					new_surface.pass = MaterialPass::Mask;
					break;
				case fastgltf::AlphaMode::Blend:
					new_surface.pass = MaterialPass::Blend;
					break;
				default:
					break;
				}
				// auto [pass_type, double_sided] = materials[idx];
				new_surface.material_id = static_cast<uint32_t>(idx + initial_materials_size);
				// new_surface.pass = pass_type;
			}
			else
			{
				// TODO: refactor - mesh has no material, assign first material
				// auto [pass_type, double_sided] = materials[0];
				// new_surface.material_id = 0;
				// new_surface.pass = pass_type;

				assert(0);
			}

			new_mesh->surfaces.push_back(new_surface);
		}
	}

	// load all nodes and their meshes
	auto node_idx = 0;
	for (fastgltf::Node& node : asset.nodes)
	{
		std::shared_ptr<Node> new_node{};

		if (node.meshIndex.has_value())
		{
			new_node = std::make_shared<Node>();
			new_node->mesh = meshes[*(node.meshIndex)];
		}
		else
		{
			// fmt::println("node has no mesh: ", node.name.c_str());
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

	for (int i = 0; i < asset.nodes.size(); i++)
	{
		fastgltf::Node& node = asset.nodes[i];
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

	// fmt::println("size of topnodes: {}", scene->top_nodes.size());
	// fmt::println("size of nodes: {}", scene->nodes.size());
	// fmt::println("size of gltf nodes: {}", asset.nodes.size());

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
}


