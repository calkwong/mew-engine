import bindings;

struct PushConstants
{
	int2 image_size;
	uint texture_id;
	uint image_id;
	uint lod;
};

[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(32, 32, 1)]
void main(
    uint3 global_id : SV_DispatchThreadID
)
{
	int2 pos = int2(global_id.xy);

	float2 uv = float2(pos) + float2(0.5);
	uv = uv / pc.image_size;

	float depth = textures[pc.texture_id].SampleLevel(samplers[DEPTH_REDUCTION_SAMPLER], uv, pc.lod).r;

	images_r32f[pc.image_id].Store(pos, depth);
}
