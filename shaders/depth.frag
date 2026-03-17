#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "bindings.glsl"
#include "buffer_references.glsl"

layout (push_constant) uniform constants
{
	mat4 view_proj;
	MaterialBuffer material_buffer;
	ObjectBuffer object_buffer;
	VertexBuffer vertex_buffer;
};

layout (location = 0) in vec2 in_uv;
layout (location = 1) flat in uint in_material_id;

// 1 - OPAQUE
// 0 - MASK
layout (constant_id = 0) const int OPAQUE = 1;

void main()
{
	if (OPAQUE == 0)
	{
		MaterialData m = material_buffer.materials[in_material_id];
			
		vec4 albedo = m.base_color_factor;
		if (m.diffuse_id != 0)
			albedo *= texture(sampler2D(textures[m.diffuse_id], samplers[LINEAR_SAMPLER]), in_uv);
			
		if (albedo.a < 0.5)
			discard;
	}
}