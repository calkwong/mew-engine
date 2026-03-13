#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "bindings.glsl"
#include "buffer_references.glsl"

layout (push_constant) uniform constants
{
	mat4 viewproj;
	MaterialBuffer materialBuffer;
//	ObjectBuffer objectBuffer;
//	VertexBuffer vertexBuffer;
	uint padding[2 * 2];
} pc;

layout (location = 0) in vec2 inUV;
layout (location = 1) flat in uint inMaterialID;

// 1 - OPAQUE
// 0 - MASK
layout (constant_id = 0) const int OPAQUE = 1;

void main()
{
	if (OPAQUE == 0)
	{
		MaterialData m = pc.materialBuffer.materials[inMaterialID];
			
		vec4 albedo = m.baseColorFactor;
		if (m.diffuseID != 0)
			albedo *= texture(sampler2D(textures[m.diffuseID], samplers[LINEAR_SAMPLER]), inUV);
			
		if (albedo.a < 0.5)
			discard;
	}
}