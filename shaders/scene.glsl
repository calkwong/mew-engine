layout(set = 0, binding = 0) uniform SceneData
{   
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	mat4 lightRot;
	mat4 shadowTransforms[4];
	vec4 cascadeSplits;
	vec4 cameraPos;
	vec4 sunlightColor;
	vec4 sunlightDir; //w for sun power
	vec4 textures;
} sceneData;