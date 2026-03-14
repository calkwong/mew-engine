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

// 10-10-10-2
vec3 decodeNormal(uint n)
{
	// can we get away without normalizing?
	return normalize(((ivec3(n) >> ivec3(20, 10, 0)) & ivec3(1023)) / 511.0 - 1.0); 
}