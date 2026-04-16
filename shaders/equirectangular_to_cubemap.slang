#version 460

#extension GL_GOOGLE_include_directive : require

#include "bindings.glsl"

layout (local_size_x = 32, local_size_y = 32) in;

#define PI 3.14159265359

layout (push_constant) uniform constants
{
	vec2 image_size;
	uint texture_id;
	uint image_id;
	float roughness;
};

vec2 sample_equirectangular_map(vec3 dir)
{
	float u = atan(dir.z, dir.x);
	float v = asin(dir.y);
	//float v = acos(dir.y); // remove 0.5 below

	u = u / (2.0 * PI) + 0.5;
	v = v / PI + 0.5;

	v = 1.0 - v;

	return vec2(u, v);
}

void main()
{
	vec3 direction = vec3(0.0);

	vec2 texel_coord = vec2(gl_GlobalInvocationID.xy);
	vec2 uv = (texel_coord + vec2(0.5)) / image_size;
	uv = uv * 2.0 - 1.0;
	uv.y *= -1.0;

	uint face = gl_WorkGroupID.z;

	switch (face)
	{
		case 0: direction = normalize(vec3(1.0, uv.y, -uv.x)); break; // +X
		case 1: direction = normalize(vec3(-1.0, uv.y, uv.x)); break; // -X
		case 2: direction = normalize(vec3(uv.x, 1.0, -uv.y)); break; // +Y
		case 3: direction = normalize(vec3(uv.x, -1.0, uv.y)); break; // -Y
		case 4: direction = normalize(vec3(uv.x, uv.y, 1.0)); break; // +Z
		case 5: direction = normalize(vec3(-uv.x, uv.y, -1.0)); break; // -Z
		default: direction = vec3(0.0); break; // Should not happen
	}

	vec2 sample_uv = sample_equirectangular_map(direction);
	vec4 color = texture(sampler2D(textures[texture_id], samplers[LINEAR_SAMPLER]), sample_uv);
	imageStore(images_array_rgba32f[image_id], ivec3(texel_coord, face), color);
}
