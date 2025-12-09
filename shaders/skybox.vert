// TODO: REFACTOR, PLENTY OF COMMITS BEHIND

#version 450

layout (location = 0) out vec3 outUVW;

layout( push_constant ) uniform constants
{
	mat4 inverseViewProj; // no translation in view matrix
	uint texture_id;
} pc;

void main()
{
	vec2 outUV = vec2(gl_VertexIndex & 2, (gl_VertexIndex << 1) & 2);
	vec4 pos = vec4(outUV.x * 2.0f - 1.0f, outUV.y * -2.0f + 1.0f, 0.0f, 1.0f); 
	gl_Position = pos;
	vec4 direction = pc.inverseViewProj * pos;
	outUVW = direction.xyz;
}