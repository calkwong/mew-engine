import bindings;

struct PushConstants
{
	float2 image_size;
	uint texture_id;
	uint image_id;
	float roughness;
};

[[vk::push_constant]] PushConstants pc;

static const float SAMPLE_DELTA = 0.025;
static const float PI = 3.14159265359;

[shader("compute")]
[numthreads(32, 32, 1)]
void main(
    uint3 global_id : SV_DispatchThreadID,
    uint3 wg_id : SV_GroupID
)
{
	float3 irradiance = float3(0.0);

	float3 direction = float3(0.0);

	int2 texel_coord = int2(global_id.xy);
	float2 uv = (float2(texel_coord) + float2(0.5)) / pc.image_size;
	uv = uv * 2.0 - 1.0;

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
	float3 up = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
	float3 T = normalize(cross(up, N));
	float3 B = cross(N, T);

	uint sample_count = 0;
	for (float phi = 0.0; phi < (PI * 2.0); phi += SAMPLE_DELTA)
	{
		for (float theta = 0.0; theta < (PI * 0.5); theta += SAMPLE_DELTA)
		{
			float3 temp_vec = cos(phi) * T + sin(phi) * B;
			float3 sample_vec = cos(theta) * N + sin(theta) * temp_vec;
			SamplerState samp = get_sampler<SamplerState>(CUBE_SAMPLER);
			irradiance += get_sampled_image<TextureCube>(pc.texture_id).SampleLevel(samp, sample_vec, 0).xyz * cos(theta) * sin(theta);
			sample_count++;
		}
	}

	irradiance = PI * irradiance / float(sample_count);
	RWTexture2DArray<float4> tex = get_storage_image<RWTexture2DArray<float4>>(pc.image_id);
	tex.Store(int3(texel_coord, face), float4(irradiance, 1.0));
}
