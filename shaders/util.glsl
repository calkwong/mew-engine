vec3 srgb_to_linear(vec3 c)
{
	return pow(c, vec3(2.2));
}

vec4 srgb_to_linear(vec4 c)
{
	return vec4(pow(c.xyz, vec3(2.2)), c.w);
}

vec3 linear_to_srgb(vec3 c)
{
	return pow(c, vec3(1.0 / 2.2));
}

vec4 linear_to_srgb(vec4 c)
{
	return vec4(pow(c.xyz, vec3(1.0 / 2.2)), c.w);
}

// 10-10-10-2
vec3 decode_normal(uint n)
{
	return normalize(((ivec3(n) >> ivec3(20, 10, 0)) & ivec3(1023)) / 511.0 - 1.0); 
}

// 8-8
vec2 decode_tangent(uint t)
{
	return ((ivec2(t) >> ivec2(8, 0)) & ivec2(255)) / 127.0 - 1.0;
}

vec2 encode_oct(vec3 n)
{
	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	vec2 s = vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
	n.xy = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * s;
	//n.xy = n.xy * 0.5 + 0.5;
	return n.xy;
}

// https://x.com/Stubbesaurus/status/937994790553227264/
vec3 decode_oct(vec2 f)
{
	vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
	float t = max(-n.z, 0.0);
	n.xy += vec2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
	return normalize(n);
}

void unpack_tbn(uint n, uint t, out vec3 normal, out vec4 tangent)
{
	normal = decode_normal(n);
	vec2 tp = decode_tangent(t);
	tangent.xyz = decode_oct(tp);
	tangent.w = (n & (1 << 30)) != 0 ? 1.0 : -1.0;
}

vec3 rotate_quat(vec3 v, vec4 q)
{
	return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

uint hash(uint a)
{
   a = (a+0x7ed55d16) + (a<<12);
   a = (a^0xc761c23c) ^ (a>>19);
   a = (a+0x165667b1) + (a<<5);
   a = (a+0xd3a2646c) ^ (a<<9);
   a = (a+0xfd7046c5) + (a<<3);
   a = (a^0xb55a4f09) ^ (a>>16);
   return a;
}