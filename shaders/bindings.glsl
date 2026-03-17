#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform Uniforms
{   
	mat4 view;
	mat4 proj;
	mat4 view_proj;
	mat4 inverse_view_proj;
	mat4 prev_view_proj;
	mat4 light_rot;
	mat4 shadow_transforms[4];
	vec4 cascade_splits;
	vec4 camera_pos;
	vec4 sunlight_color;
	vec4 sunlight_dir; //w for sun power
	vec4 textures; // cubemap/skybox, irradiance, prefiltered, brdf
 	mat4 shadow_views[4]; // for shadow cull
	vec4 shadow_widths; // for shadow cull, each channel for a diff cascade; TODO: move to pc?
} uniforms;

layout(set = 1, binding = 0, r32f) uniform image2D images_r32f[];
layout(set = 1, binding = 0, rgba32f) uniform image2DArray images_array_rgba32f[];
layout(set = 1, binding = 0, rgba32f) uniform image2D images_rgba32f[];
layout(set = 1, binding = 0, rg16f) uniform image2D images_rg16f[];

layout(set = 2, binding = 0) uniform texture2D textures[];
layout(set = 2, binding = 0) uniform textureCube textures_cube[];
layout(set = 2, binding = 0) uniform utexture2D textures_u[];

#define LINEAR_SAMPLER 0
#define CUBE_SAMPLER 1
#define SHADOW_SAMPLER 2
#define DEPTH_REDUCTION_SAMPLER 3
#define NEAREST_CLAMP_BORDER_SAMPLER 4
#define NEAREST_SAMPLER 5 // clamp to edge
#define LINEAR_CLAMP_SAMPLER 6 // clamp to edge

layout(set = 3, binding = 0) uniform sampler samplers[];