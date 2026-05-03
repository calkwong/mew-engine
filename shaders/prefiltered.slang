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

static const float PI = 3.14159265359;

[shader("compute")]
[numthreads(32, 32, 1)]
void main(
    uint3 global_id : SV_DispatchThreadID,
    uint3 wg_id : SV_GroupID
)
{
	float3 direction = float3(0.0);

	int2 texel_coord = int2(global_id.xy);
	int2 image_size;
	int layers;
	images_array_rgba32f[pc.image_id].GetDimensions(image_size.x, image_size.y, layers);

	if (texel_coord.x >= image_size.x || texel_coord.y >= image_size.y)
		return;

	float2 uv = (float2(texel_coord) + float2(0.5)) / float2(image_size);
	uv = uv * 2.0 - 1.0;

	float perceptual_roughness = pc.roughness;
	int2 envmap_size;
	textures_cube[pc.texture_id].GetDimensions(0, envmap_size.x, envmap_size.y, layers);

	uint face = wg_id.z;

	switch (face)
	{
		case 0: direction = normalize(float3(1.0, -uv.y, -uv.x)); break; // +X
		case 1: direction = normalize(float3(-1.0, -uv.y, uv.x)); break; // -X
		case 2: direction = normalize(float3(uv.x, 1.0, uv.y)); break; // +Y
		case 3: direction = normalize(float3(uv.x, -1.0, -uv.y)); break; // -Y
		case 4: direction = normalize(float3(uv.x, -uv.y, 1.0)); break; // +Z
		case 5: direction = normalize(float3(-uv.x, -uv.y, -1.0)); break; // -Z
		default: direction = float3(0.0); break;
	}

	float3 N = direction;
	float3 V = N;
	float linear_roughness = perceptual_roughness * perceptual_roughness;

	uint sample_count = 32;
	float total_weight = 0.0;
	float3 prefiltered_color = float3(0.0);
	for (uint i = 0; i < sample_count; i++)
	{
		float2 Xi = Hammersley(i, sample_count);
		float3 H = ImportanceSampleGGX(Xi, N, linear_roughness);
		float3 L = normalize(2.0 * dot(V,H) * H - V);

		float NdotL = max(dot(N, L), 0.0);
		float NdotH = max(dot(N, H), 0.0);
		if (NdotL > 0.0)
		{
			// https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
			// https://chetanjags.wordpress.com/2015/08/26/image-based-lighting/
			float pdf = D_GGX(NdotH, linear_roughness) / 4.0 + 0.0001; // NdotH and VdotH (from Jacobian) cancel out
			float omegaS = 1.0 / (float(sample_count) * pdf);  // solid angle of a sample
			float omegaP = 4.0 * PI / (6.0 * envmap_size.x * envmap_size.y); // solid angle of one pixel from cubemap/sphere
			float mipLevel = perceptual_roughness == 0.0 ? 0.0 : max(0.5 * log2(omegaS / omegaP) + 1.0, 0.0);

			prefiltered_color += textures_cube[pc.texture_id].SampleLevel(samplers[CUBE_SAMPLER], L, mipLevel).rgb * NdotL;
			total_weight += NdotL;
		}
	}

	prefiltered_color = prefiltered_color / total_weight;

	images_array_rgba32f[pc.image_id].Store(int3(texel_coord, face), float4(prefiltered_color, 1.0));
}
