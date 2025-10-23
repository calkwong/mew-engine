layout(set = 0, binding = 0) uniform SceneData
{   
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	mat4 shadowTransform;
	vec3 cameraPos;
	uint irradiance_id;
	vec4 sunlightColor;
	vec4 sunlightDir; //w for sun power
	uint prefiltered_id;
	uint brdf_id;
	uint shadow_id;
} sceneData;