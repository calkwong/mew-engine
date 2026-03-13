vec3 srgbToLinear(vec3 c)
{
	return pow(c, vec3(2.2));
}

vec4 srgbToLinear(vec4 c)
{
	return vec4(pow(c.xyz, vec3(2.2)), c.w);
}

vec3 linearToSrgb(vec3 c)
{
	return pow(c, vec3(1.0 / 2.2));
}

vec4 linearToSrgb(vec4 c)
{
	return vec4(pow(c.xyz, vec3(1.0 / 2.2)), c.w);
}