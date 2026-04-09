#version 460

#extension GL_GOOGLE_include_directive : require

#include "bindings.glsl"

layout (local_size_x = 32, local_size_y = 32) in;

layout(push_constant) uniform constants
{
	ivec2 image_size;
	uint texture_id;
	uint image_id;
	uint lod;
};

void main()
{
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

	vec2 uv = vec2(pos) + vec2(0.5);
	uv = uv / image_size;

	float depth = textureLod(sampler2D(textures[texture_id], samplers[DEPTH_REDUCTION_SAMPLER]), uv, lod).r;

	imageStore(images_r32f[image_id], pos, vec4(depth));
}
