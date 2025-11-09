#include <vk_scene.h>
#include <vk_engine.h>

#include <vulkan/vulkan.h>

#include <vector>

std::vector<IndirectBatch> build_indirect_array(const std::vector<RenderObject>& renderables)
{
	std::vector<IndirectBatch> batches{};

	std::shared_ptr<MeshAsset> last_mesh{};
	ShaderPass* last_material{};

	for (size_t i = 0; i < renderables.size(); i++)
	{
		auto obj = renderables[i];

		bool same_mesh = obj.mesh == last_mesh;
		bool same_material = obj.material->forward_pass == last_material;

		if (same_mesh && same_material)
			batches.back().count++;
		else
		{
			last_mesh = obj.mesh;
			last_material = obj.material->forward_pass;

			IndirectBatch new_batch{};
			new_batch.mesh = last_mesh;
			new_batch.forward_pass = last_material;
			new_batch.first = i;
			new_batch.count = 1;
			batches.push_back(new_batch);
		}
	}

	return batches;
}

std::vector<MultiBatch> build_multibatch_array(std::vector<IndirectBatch>& batches)
{
	std::vector<MultiBatch> multibatches{};

	std::shared_ptr<MeshAsset> last_mesh{};
	ShaderPass* last_material{};

	multibatches.emplace_back(MultiBatch{ 0, 0 });

	for (size_t i = 0; i < batches.size(); i++)
	{
		const auto& batch = batches[i];
		bool same_mesh = last_mesh == batch.mesh;
		bool same_material = last_material == batch.forward_pass;

		if (same_mesh && same_material)
			multibatches.back().count++;
		else
		{
			multibatches.emplace_back(MultiBatch{ static_cast<uint32_t>(i), 1 });
		}
	}

	return multibatches;
}