#version 450

#extension GL_EXT_nonuniform_qualifier : require

layout (location = 0) in vec2 inUV;

layout(set = 0, binding = 0) uniform texture2D allTextures[];
layout(set = 1, binding = 0) uniform sampler samplers[];

layout (location = 0) out vec4 outFragColor;

const vec2 invAtan = vec2(0.1591, 0.3183);

// push constant block
layout (push_constant) uniform constants
{
	uint face;
	uint texture_id;
} pc;

vec2 sampleSphericalMap(vec3 v)
{
	vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
	uv *= invAtan;
	uv += 0.5;
	uv.y = 1.0 - uv.y;
	return uv;
}

void main()
{
	vec3 direction;

    vec2 uv = inUV * 2.0 - 1.0; // Map to range [-1, 1]

	switch (pc.face)
	{
		case 0: direction = normalize(vec3(1.0, -uv.y, -uv.x)); break; // +X
		case 1: direction = normalize(vec3(-1.0, -uv.y, uv.x)); break; // -X
		case 2: direction = normalize(vec3(uv.x, 1.0, uv.y)); break; // +Y
		case 3: direction = normalize(vec3(uv.x, -1.0, -uv.y)); break; // -Y
		case 4: direction = normalize(vec3(uv.x, -uv.y, 1.0)); break; // +Z
		case 5: direction = normalize(vec3(-uv.x, -uv.y, -1.0)); break; // -Z
		default: direction = vec3(0.0); break; // Should not happen
	}

	uv = sampleSphericalMap(direction);
	vec3 color = texture(sampler2D(allTextures[pc.texture_id], samplers[0]), uv).rgb;

	outFragColor = vec4(color, 1.0);
}