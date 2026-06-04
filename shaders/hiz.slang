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
	SamplerState samp = get_sampler<SamplerState>(DEPTH_REDUCTION_SAMPLER);
	float depth = get_sampled_image<Texture2D>(pc.texture_id).SampleLevel(samp, uv, pc.lod).r;
	RWTexture2D<float> tex = get_storage_image<RWTexture2D<float>>(pc.image_id);
	tex.Store(pos, depth);
}
