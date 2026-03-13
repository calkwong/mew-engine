#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform SceneData
{   
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	mat4 inverseViewproj;
	mat4 previousViewproj; 
	mat4 lightRot; 
	mat4 shadowTransforms[4];
	vec4 cascadeSplits;
	vec4 cameraPos;
	vec4 sunlightColor;
	vec4 sunlightDir; //w for sun power
	vec4 textures; // cubemap/skybox, irradiance, prefiltered, brdf
 	mat4 shadowViews[4]; // for shadow cull
	vec4 shadowWidths; // for shadow cull, each channel for a diff cascade; TODO: move to pc?
} sceneData;

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