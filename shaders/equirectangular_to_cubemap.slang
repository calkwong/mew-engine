import bindings;

struct PushConstants
{
	float2 image_size;
	uint texture_id;
	uint image_id;
};

[[vk::push_constant]] PushConstants pc;

static const float PI = 3.14159265359;

float2 sample_equirectangular_map(float3 dir)
{
	float u = atan2(dir.z, dir.x);
	float v = asin(dir.y);
	//float v = acos(dir.y); // remove 0.5 below

	u = u / (2.0 * PI) + 0.5;
	v = v / PI + 0.5;

	v = 1.0 - v;

	return float2(u, v);
}

[shader("compute")]
[numthreads(32, 32, 1)]
void main(
    uint3 global_id : SV_DispatchThreadID,
    uint3 wg_id : SV_GroupID
)
{
	float3 direction = float3(0.0);

	int2 texel_coord = int2(global_id.xy);
	float2 uv = (float2(texel_coord) + float2(0.5)) / pc.image_size;
	uv = uv * 2.0 - 1.0;
	uv.y *= -1.0;

	uint face = wg_id.z;

	switch (face)
	{
		case 0: direction = normalize(float3(1.0, uv.y, -uv.x)); break; // +X
		case 1: direction = normalize(float3(-1.0, uv.y, uv.x)); break; // -X
		case 2: direction = normalize(float3(uv.x, 1.0, -uv.y)); break; // +Y
		case 3: direction = normalize(float3(uv.x, -1.0, uv.y)); break; // -Y
		case 4: direction = normalize(float3(uv.x, uv.y, 1.0)); break; // +Z
		case 5: direction = normalize(float3(-uv.x, uv.y, -1.0)); break; // -Z
		default: direction = float3(0.0); break; // Should not happen
	}

	float2 sample_uv = sample_equirectangular_map(direction);
	SamplerState samp = get_sampler<SamplerState>(LINEAR_SAMPLER);
	float4 color = get_sampled_image<Texture2D>(pc.texture_id).SampleLevel(samp, sample_uv, 0.0);
	RWTexture2DArray<float4> tex = get_storage_image<RWTexture2DArray<float4>>(pc.image_id);
	tex.Store(int3(texel_coord, face), color);
}
