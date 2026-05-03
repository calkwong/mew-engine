import bindings;
import sampling;
import pbr;

struct PushConstants
{
	float2 image_size;
	uint texture_id;
	uint image_id;
	float roughness;
};

[[vk::push_constant]] PushConstants pc;

float2 IntegratedBRDF(float NdotV, float perceptual_roughness)
{
	float3 V = float3(0.0);
	V.x = sqrt(1.0 - NdotV * NdotV);
    V.y = 0.0;
    V.z = NdotV;

	float3 N = float3(0.0, 0.0, 1.0);
	float linear_roughness = perceptual_roughness * perceptual_roughness;

	float scale = 0.0;
	float bias = 0.0;

	uint sample_count = 1024;
	for (uint i = 0; i < sample_count; i++)
	{
		float2 Xi = Hammersley(i, sample_count);
		float3 H = ImportanceSampleGGX(Xi, N, linear_roughness);
		float3 L = normalize(2.0 * dot(V,H) * H - V);

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

	return 4.0 * float2(scale, bias) / float(sample_count);
}

[shader("compute")]
[numthreads(32, 32, 1)]
void main(
    uint3 global_id : SV_DispatchThreadID
)
{
	float2 texel_coord = float2(global_id.xy);
	float2 uv = (texel_coord + float2(0.5)) / pc.image_size;

	float2 color = IntegratedBRDF(uv.x, uv.y);

	images_rg16f[pc.image_id].Store(int2(texel_coord), color);
}
