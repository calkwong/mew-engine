#version 460

#extension GL_GOOGLE_include_directive : require

#include "bindings.glsl"
#include "importance_sampling.glsl"
#include "pbr.glsl"

layout (local_size_x = 32, local_size_y = 32) in;

layout (push_constant) uniform constants
{
	vec2 image_size;
	uint texture_id;
	uint image_id;
	float roughness;
};

vec2 IntegratedBRDF(float NdotV, float perceptual_roughness)
{
	vec3 V = vec3(0.0);
	V.x = sqrt(1.0 - NdotV * NdotV);
    V.y = 0.0;
    V.z = NdotV;

	vec3 N = vec3(0.0, 0.0, 1.0);
	float linear_roughness = perceptual_roughness * perceptual_roughness;

	float scale = 0.0;
	float bias = 0.0;

	uint sample_count = 1024;
	for (uint i = 0; i < sample_count; i++)
	{
		vec2 Xi = Hammersley(i, sample_count);
		vec3 H = ImportanceSampleGGX(Xi, N, linear_roughness);
		vec3 L = normalize(2.0 * dot(V,H) * H - V);

		float NdotL = max(L.z, 0.0);
		float NdotH = max(H.z, 0.0);
		float VdotH = max(dot(V, H), 0.0);

		if (NdotL > 0.0)
		{
			float V_pdf = V_SmithGGXCorrelated(NdotV, NdotL, linear_roughness) * VdotH * NdotL / NdotH;
			float Fc = pow(1.0 - VdotH, 5.0);
			scale += V_pdf * (1.0 - Fc);
			bias += V_pdf * Fc;
		}
	}

	return 4.0 * vec2(scale, bias) / float(sample_count);
}

void main()
{
	vec2 texel_coord = vec2(gl_GlobalInvocationID.xy);
	vec2 uv = (texel_coord + vec2(0.5)) / image_size;

	vec2 color = IntegratedBRDF(uv.x, uv.y);

	imageStore(images_rg16f[image_id], ivec2(texel_coord), vec4(color, 0.0, 1.0));
}
