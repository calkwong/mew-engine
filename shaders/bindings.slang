#ifndef BINDINGS_SLANG
#define BINDINGS_SLANG

struct UniformBuffer
{
    float4x4 view;
	float4x4 proj;
	float4x4 view_proj;
	float4x4 inverse_view_proj;
	float4x4 prev_view_proj;
	float4x4 light_rot;
	float4x4 shadow_transforms[4];
	float4 cascade_splits;
	float4 camera_pos;
	float4 sunlight_color;
	float4 sunlight_dir; //w for sun power
	float4 textures; // cubemap/skybox, irradiance, prefiltered, brdf
	float4x4 shadow_views[4]; // for shadow cull
	float4 shadow_widths; // for shadow cull, each channel for a diff cascade; TODO: move to pc?
}

layout(set = 0, binding = 0) ConstantBuffer<UniformBuffer> uniforms;

layout(set = 1, binding = 0, r32f) RWTexture2D<float4> images_r32f[];
layout(set = 1, binding = 0, rgba32f) RWTexture2DArray<float4> images_array_rgba32f[];
layout(set = 1, binding = 0, rgba32f) RWTexture2D<float4> images_rgba32f[];
layout(set = 1, binding = 0, rg16f) RWTexture2D<float4> images_rg16f[];

layout(set = 2, binding = 0) Texture2D<float4> textures[];
layout(set = 2, binding = 0) TextureCube<float4> textures_cube[];
layout(set = 2, binding = 0) Texture2D<uint4> textures_u[];

#define LINEAR_SAMPLER 0
#define CUBE_SAMPLER 1
#define SHADOW_SAMPLER 2
#define DEPTH_REDUCTION_SAMPLER 3
#define NEAREST_CLAMP_BORDER_SAMPLER 4
#define NEAREST_SAMPLER 5 // clamp to edge
#define LINEAR_CLAMP_SAMPLER 6 // clamp to edge

layout(set = 3, binding = 0) SamplerState samplers[];

#endif
