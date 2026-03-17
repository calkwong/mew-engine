#version 450

#extension GL_GOOGLE_include_directive : require

#include "bindings.glsl"

layout (location = 0) out vec2 out_uv;

void main()
{
	out_uv = vec2(gl_VertexIndex & 2, (gl_VertexIndex << 1) & 2);
	gl_Position = vec4(out_uv.x * 2.0f - 1.0f, out_uv.y * -2.0f + 1.0f, 0.0f, 1.0f);
}