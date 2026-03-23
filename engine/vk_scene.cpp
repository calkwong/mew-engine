#include "vk_scene.h"

// POST MESH SHADERS: do we still need this?
void RenderScene::init()
{
	opaque_pass.type = MeshPassType::Opaque;
	mask_pass.type = MeshPassType::Mask;
	transparent_pass.type = MeshPassType::Transparent;
}
