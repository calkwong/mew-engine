#include "common.h"
#include "vk_scene.h"
#include "vk_math.h"
#include "resources.h"

#include <array>
#include <vector>

// POST MESH SHADERS: do we still need this?
void RenderScene::init()
{
	opaque_pass.type = MeshPassType::Opaque;
	mask_pass.type = MeshPassType::Mask;
	transparent_pass.type = MeshPassType::Transparent;
}