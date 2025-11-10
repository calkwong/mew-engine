#include <vk_loader.h>

#include "vk_engine.h"
#include "vk_types.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "mikktspace.h"

#include <vulkan/vulkan.h>
#include <glm/gtx/quaternion.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>
#include <fmt/core.h>

#include <limits>
#include <optional>
#include <vector>

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
	switch (filter) {
	case fastgltf::Filter::NearestMipMapNearest:
	case fastgltf::Filter::LinearMipMapNearest:
		return VK_SAMPLER_MIPMAP_MODE_NEAREST;

	case fastgltf::Filter::NearestMipMapLinear:
	case fastgltf::Filter::LinearMipMapLinear:
	default:
		return VK_SAMPLER_MIPMAP_MODE_LINEAR;
	}
}

std::optional<AllocatedImage> load_image(VulkanEngine* engine, fastgltf::Asset& asset, fastgltf::Image& image, VkFormat format, bool mipmapped = false)
{
	AllocatedImage new_image{};

	int width{};
	int height{};
	int channels{};

	std::visit(
		fastgltf::visitor{
			[](auto& arg) {},
			[&](fastgltf::sources::URI& filePath) {
					assert(filePath.fileByteOffset == 0); // we don't support offsets with stbi
					assert(filePath.uri.isLocalPath()); // only capable of loading local files

					const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());

					//std::string current_path = "../../assets/khronos_sponza/" + path; // (!) TODO handle this properly
					std::string current_path = "../../assets/bistro_interior/" + path; // (!) TODO handle this properly
					//std::string current_path = "../../assets/bistro_exterior/" + path; // (!) TODO handle this properly
					unsigned char* data = stbi_load(current_path.c_str(), &width, &height, &channels, 4);
					if (data)
					{
						VkExtent3D image_size{ static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
						new_image = engine->create_image(data, image_size, format, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, mipmapped);

						stbi_image_free(data);
					}
				},
			[&](fastgltf::sources::Vector& vector) {
					unsigned char* data = stbi_load_from_memory(vector.bytes.data(), static_cast<int>(vector.bytes.size()), &width, &height, &channels, 4);
					if (data)
					{
						VkExtent3D image_size{ static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
						new_image = engine->create_image(data, image_size, format, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, mipmapped);

						stbi_image_free(data);
					}
				},
			[&](fastgltf::sources::BufferView& view) {
					auto& bufferView = asset.bufferViews[view.bufferViewIndex];
					auto& buffer = asset.buffers[bufferView.bufferIndex];
					std::visit(fastgltf::visitor{
						[](auto& arg) {},
						[&](fastgltf::sources::Vector& vector) {
								unsigned char* data = stbi_load_from_memory(vector.bytes.data() + bufferView.byteOffset, 
									static_cast<int>(bufferView.byteLength), &width, &height, &channels, 4
								);

								if (data)
								{
									VkExtent3D image_size{ static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };

									new_image = engine->create_image(data, image_size, format, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, mipmapped);

									stbi_image_free(data);
								}
							},
						},
						buffer.data
					);
				}
		},
		image.data
	);

	// if any attempts of the above to load image data failed, we haven't written the image
	// so handle is null
	if (new_image.image == VK_NULL_HANDLE)
	{
		fmt::println("load image error");
		return {};
	}
	return new_image;
}

std::optional<std::shared_ptr<LoadedGLTF>> load_gltf(VulkanEngine* engine, std::string_view file_path)
{
	fmt::println("Loading GLTF: {}", file_path);

	std::shared_ptr<LoadedGLTF> scene = std::make_shared<LoadedGLTF>();
	scene->creator = engine;
	LoadedGLTF& file = *scene;

	// testing extensions
	constexpr auto enabled_extensions = fastgltf::Extensions::KHR_lights_punctual;

	fastgltf::Parser parser(enabled_extensions);
	//fastgltf::Parser parser{};

	constexpr auto gltf_options{
		fastgltf::Options::DontRequireValidAssetMember |
		fastgltf::Options::LoadGLBBuffers |
		fastgltf::Options::AllowDouble |
		fastgltf::Options::LoadExternalBuffers
	};

	fastgltf::GltfDataBuffer data{};
	data.loadFromFile(file_path);

	fastgltf::Asset gltf{};

	std::filesystem::path path{ file_path };

	auto type = fastgltf::determineGltfFileType(&data);
	if (type == fastgltf::GltfType::glTF)
	{
		auto load = parser.loadGLTF(&data, path.parent_path(), gltf_options);
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
		auto load{ parser.loadBinaryGLTF(&data, path.parent_path(), gltf_options) };
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

	// (!) load samplers - not used, to handle
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

		VkSampler sampler{};
		vkCreateSampler(engine->device, &info, nullptr, &sampler);

		file.samplers.push_back(sampler);
	}

	std::vector<std::shared_ptr<MeshAsset>> meshes{};
	std::vector<std::shared_ptr<Node>> nodes{};
	std::vector<MaterialInfo> materials{}; // id

	// image loading deferred to material creation to prevent performance overhead from image_create_mutable_bit
	fmt::println("gltf file has {} images", gltf.images.size());
	std::vector<AllocatedImage> images(gltf.images.size());
	std::vector<bool> images_set(gltf.images.size());

	//for (fastgltf::Image& image : gltf.images)
	//	fmt::println("image: {}", image.name.c_str()); // debug

	fmt::println("gltf file has {} materials", gltf.materials.size());
	const size_t materials_size = (gltf.materials.size() > 0) ? gltf.materials.size() : 1; // default material fallback

	file.material_buffer = engine->create_buffer(
		materials_size * sizeof(MaterialData), VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
	);

	MaterialData* scene_material_data{};
	scene_material_data = static_cast<MaterialData*>(file.material_buffer.info.pMappedData);

	if (gltf.materials.size() == 0)
	{
		MaterialData mat_data{};
		scene_material_data[0] = mat_data;
		materials.emplace_back(MaterialInfo{MaterialPass::Opaque, 0});
	}

	VkBufferDeviceAddressInfo address_info{};
	address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	address_info.buffer = file.material_buffer.buffer;

	file.material_buffer_address = vkGetBufferDeviceAddress(engine->device, &address_info);

	// need to implement MaterialCache as its common for gltf to have same material under different name
	// current implementation simply duplicates this in the material buffer
	int material_idx{ 0 };
	for (fastgltf::Material& mat : gltf.materials)
	{
		//fmt::println("material: {}", mat.name.c_str()); // debug

		MaterialData mat_data{};
		mat_data.base_color_factor.x = mat.pbrData.baseColorFactor[0];
		mat_data.base_color_factor.y = mat.pbrData.baseColorFactor[1];
		mat_data.base_color_factor.z = mat.pbrData.baseColorFactor[2];
		mat_data.base_color_factor.w = mat.pbrData.baseColorFactor[3];
		mat_data.metallic_factor = mat.pbrData.metallicFactor;
		mat_data.roughness_factor = mat.pbrData.roughnessFactor;
		
		if (mat.pbrData.baseColorTexture.has_value())
		{
			size_t idx = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
			//size_t sampler = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].samplerIndex.value();
			
			std::optional<AllocatedImage> img{};
			if (!images_set[idx])
			{
				images_set[idx] = true;
				img = load_image(engine, gltf, gltf.images[idx], VK_FORMAT_R8G8B8A8_SRGB, true); 
				if (img.has_value())
				{
					images[idx] = (*img);
					file.images[std::to_string(idx).c_str()] = images[idx];
				}
				else
				{
					images.push_back(engine->error_image); 
					img = engine->error_image;
				}
			}
			else
			{
				img = images[idx];
			}
			mat_data.diffuse_id = engine->texture_cache.add_texture(img.value().view); // img guaranteed to have value
		}

		if (mat.pbrData.metallicRoughnessTexture.has_value())
		{
			size_t idx = gltf.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].imageIndex.value();
			//size_t sampler{ gltf.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].samplerIndex.value() };
			
			std::optional<AllocatedImage> img{};
			if (!images_set[idx])
			{
				images_set[idx] = true;
				img = load_image(engine, gltf, gltf.images[idx], VK_FORMAT_R8G8B8A8_UNORM, true);
				if (img.has_value())
				{
					images[idx] = (*img);
					file.images[std::to_string(idx).c_str()] = images[idx];
				}
				else
				{
					images.push_back(engine->error_image);
					img = engine->error_image;
				}
			}
			else
			{
				img = images[idx];
			}
			mat_data.metal_roughness_id = engine->texture_cache.add_texture(img.value().view);
		}

		if (mat.normalTexture.has_value())
		{
			size_t idx = gltf.textures[mat.normalTexture.value().textureIndex].imageIndex.value();
			std::optional<AllocatedImage> img{};
			if (!images_set[idx])
			{
				images_set[idx] = true;
				img = load_image(engine, gltf, gltf.images[idx], VK_FORMAT_R8G8B8A8_UNORM, true);
				if (img.has_value())
				{
					images[idx] = (*img);
					file.images[std::to_string(idx).c_str()] = images[idx];
				}
				else
				{
					images.push_back(engine->error_image);
					img = engine->error_image;
				}
			}
			else
			{
				img = images[idx];
			}
			mat_data.normal_id = engine->texture_cache.add_texture(img.value().view);
		}

		if (mat.occlusionTexture.has_value())
		{
			size_t idx = gltf.textures[mat.occlusionTexture.value().textureIndex].imageIndex.value();
			std::optional<AllocatedImage> img{};
			if (!images_set[idx])
			{
				images_set[idx] = true;
				img = load_image(engine, gltf, gltf.images[idx], VK_FORMAT_R8G8B8A8_UNORM, true);
				if (img.has_value())
				{
					images[idx] = (*img);
					file.images[std::to_string(idx).c_str()] = images[idx];
				}
				else
				{
					images.push_back(engine->error_image);
					img = engine->error_image;
				}
			}
			else
			{
				img = images[idx];
			}
			mat_data.occlusion_id = engine->texture_cache.add_texture(img.value().view);
		}

		if (mat.emissiveTexture.has_value())
		{
			size_t idx = gltf.textures[mat.emissiveTexture.value().textureIndex].imageIndex.value();
			std::optional<AllocatedImage> img{};
			if (!images_set[idx])
			{
				images_set[idx] = true;
				img = load_image(engine, gltf, gltf.images[idx], VK_FORMAT_R8G8B8A8_SRGB, true);
				if (img.has_value())
				{
					images[idx] = (*img);
					file.images[std::to_string(idx).c_str()] = images[idx];
				}
				else
				{
					images.push_back(engine->error_image);
					img = engine->error_image;
				}
			}
			else
			{
				img = images[idx];
			}
			mat_data.emissive_id = engine->texture_cache.add_texture(img.value().view);
		}
		
		scene_material_data[material_idx] = mat_data;

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

		uint32_t double_sided = static_cast<uint32_t>(mat.doubleSided);

		materials.emplace_back(MaterialInfo{ pass_type, double_sided });
		material_idx++;
	}

	std::vector<uint32_t> indices{};
	std::vector<Vertex> vertices{};

	fmt::println("gltf file has {} meshes", gltf.meshes.size());

	auto mesh_idx = 0;
	for (fastgltf::Mesh& mesh : gltf.meshes)
	{
		std::shared_ptr<MeshAsset> new_mesh{ std::make_shared<MeshAsset>() };
		meshes.push_back(new_mesh);
		//file.meshes[mesh.name.c_str()] = new_mesh;
		file.meshes[std::to_string(mesh_idx).c_str()] = new_mesh;
		mesh_idx++;
		new_mesh->name = mesh.name;
		new_mesh->material_buffer_address = file.material_buffer_address;

		indices.clear();
		vertices.clear();

		for (auto&& p : mesh.primitives)
		{
			GeoSurface new_surface{};
			new_surface.start_index = static_cast<uint32_t>(indices.size());
			new_surface.count = static_cast<uint32_t>(gltf.accessors[p.indicesAccessor.value()].count);

			size_t initial_vtx = vertices.size();

			// load indexes
			{
				fastgltf::Accessor& index_accessor = gltf.accessors[p.indicesAccessor.value()];
				indices.reserve(indices.size() + index_accessor.count);

				fastgltf::iterateAccessor<std::uint32_t>(gltf, index_accessor,
					[&](std::uint32_t idx) {
						indices.push_back(idx + static_cast<uint32_t>(initial_vtx));
					});
			}

			glm::vec3 min_pos = glm::vec3(std::numeric_limits<float>::max());
			glm::vec3 max_pos = glm::vec3(std::numeric_limits<float>::lowest());

			// load vertex positions
			{
				fastgltf::Accessor& pos_accessor = gltf.accessors[p.findAttribute("POSITION")->second];
				vertices.resize(vertices.size() + pos_accessor.count);

				fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, pos_accessor,
					[&](glm::vec3 v, size_t index) {
						Vertex new_vtx{};
						new_vtx.position = v;
						min_pos = glm::min(min_pos, v);
						max_pos = glm::max(max_pos, v);
						vertices[initial_vtx + index] = new_vtx;
					});
			}

			new_surface.bounds.origin = (max_pos + min_pos) / 2.0f;
			new_surface.bounds.extents = (max_pos - min_pos) / 2.0f;
			//new_surface.bounds.sphere_radius = glm::length(new_surface.bounds.extents);
			
			// load vertex normals
			{
				auto normals = p.findAttribute("NORMAL");
				if (normals != p.attributes.end())
				{
					fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[(*normals).second],
						[&](glm::vec3 v, size_t index) {
							vertices[initial_vtx + index].normal = v;
						});
				}
			}

			bool generate_tangents{};
			// load vertex tangents
			{
				auto tangents = p.findAttribute("TANGENT");
				if (tangents != p.attributes.end())
				{
					fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[(*tangents).second],
						[&](glm::vec4 v, size_t index) {
							vertices[initial_vtx + index].tangent = v;
						});
				}
				else
				{
					generate_tangents = true;
				}
			}

			// mikk tangent generation
			if (generate_tangents)
			{
				MikkMesh mesh{ &vertices, &indices };
				calculateTangents(mesh);
			}

			// load uvs
			{
				auto uv = p.findAttribute("TEXCOORD_0");
				if (uv != p.attributes.end())
				{
					fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).second],
						[&](glm::vec2 v, size_t index) {
							vertices[initial_vtx + index].uv_x = v.x;
							vertices[initial_vtx + index].uv_y = v.y;
						});
				}
			}

			if (p.materialIndex.has_value())
			{
				size_t idx = p.materialIndex.value();
				MaterialInfo m = materials[idx];
				new_surface.material_id = static_cast<uint32_t>(idx);
				new_surface.pass = m.pass_type;

				ShaderPass* forward{};
				ShaderPass* shadow{};
				switch (new_surface.pass)
				{
				case MaterialPass::Mask: // assumes double-sided 
					forward = engine->shader_passes["textured_lit_clip"].get();
					shadow = engine->shader_passes["shadow_flat"].get();
					break;
				case MaterialPass::Blend:
					forward = engine->shader_passes["blend"].get();
					//forward = m.double_sided ? engine->shader_passes["textured_lit2"].get() : engine->shader_passes["textured_lit"].get();
					shadow = nullptr; // transparent objs don't cast shadows for now
					//shadow = engine->shader_passes["shadow_flat"].get();
					break;
				case MaterialPass::Opaque:
					forward = m.double_sided ? engine->shader_passes["textured_lit2"].get() : engine->shader_passes["textured_lit"].get();
					shadow = m.double_sided ? engine->shader_passes["shadow_flat"].get() : engine->shader_passes["shadow"].get();
					break;
				default:
					break;
				}
				new_surface.material = engine->material_cache.add_material(forward, shadow);
			}
			else
			{
				// (!) mesh has no material, assign first material
				auto m = materials[0];
				new_surface.material_id = 0;
				ShaderPass* forward = engine->shader_passes["textured_lit"].get();
				ShaderPass* shadow = engine->shader_passes["shadow"].get();
				new_surface.material = engine->material_cache.add_material(forward, shadow);
				new_surface.pass = m.pass_type;
			}

			new_mesh->surfaces.push_back(new_surface);
		}

		new_mesh->mesh_buffer = engine->upload_mesh(indices, vertices);
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
			new_node = std::make_shared<Node>(); // (!) absorbing allocation cost for dummy node
		}

		nodes.push_back(new_node);
		file.nodes[std::to_string(node_idx).c_str()] = new_node;
		node_idx++;

		std::visit(fastgltf::visitor{
				[&](fastgltf::Node::TransformMatrix matrix) {
					memcpy(&new_node->local_transform, matrix.data(), sizeof(matrix));
				},
				[&](fastgltf::Node::TRS transform) {
					glm::vec3 tl(transform.translation[0], transform.translation[1],
						transform.translation[2]);
					glm::quat rot(transform.rotation[3], transform.rotation[0], transform.rotation[1],
						transform.rotation[2]);
					glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);

					glm::mat4 tm = glm::translate(glm::mat4(1.f), tl);
					glm::mat4 rm = glm::toMat4(rot);
					glm::mat4 sm = glm::scale(glm::mat4(1.f), sc);

					new_node->local_transform = tm * rm * sm;
				}
			},
			node.transform);
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
	VkDevice device = creator->device;

	creator->destroy_buffer(material_buffer);

	for (auto& [k, v] : meshes)
	{
		creator->destroy_buffer(v->mesh_buffer.vertex_buffer);
		creator->destroy_buffer(v->mesh_buffer.index_buffer);
	}

	for (auto& [k, v] : images)
	{
		creator->destroy_image(v); 
	}

	for (auto& s : samplers)
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
