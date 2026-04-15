import bindings;
import util;

struct PushConstants
{
	float* luminance_avg_buffer;
	float2 screen_size;
	uint src_id;
	uint dst_id;
	uint autoexpose;
	uint tonemap_func;
};

[[vk::push_constant]] PushConstants pc;

// http://filmicworlds.com/blog/filmic-tonemapping-operators/
float3 filmic_optimized(float3 c)
{
	float3 x = max(float3(0.0), c - 0.004);
	float3 color = (x * ( 6.2 * x + 0.5)) / (x * (6.2 * x +1.7) + 0.06);

	// already non-linear
	return color;
}

float3 reinhard(float3 c)
{
	return c / (float3(1.0) + c);
}

float3 reverse_reinhard(float3 c)
{
    return c / (float3(1.0) - c);
}

// Minimal AgX implementation: https://iolite-engine.com/blog_posts/minimal_agx_implementation
float3 agx_default_contrast_approx(float3 x)
{
    float3 x2 = x * x;
    float3 x4 = x2 * x2;

    return + 15.5     * x4 * x2
           - 40.14    * x4 * x
           + 31.96    * x4
           - 6.868    * x2 * x
           + 0.4298   * x2
           + 0.1191   * x
           - 0.00232;
}

float3 agx(float3 val)
{
    const float3x3 agx_mat = float3x3(
        0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0784335999999992,  0.878468636469772,  0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104
    );

    const float min_ev = -12.47393;
    const float max_ev = 4.026069;

    // input transform
    val = mul(agx_mat, val);

    // log2 space encoding
    val = clamp(log2(val), min_ev, max_ev);
    val = (val - min_ev) / (max_ev - min_ev);

    // apply sigmoid function approximation
    val = agx_default_contrast_approx(val);

    return val;
}

float3 agx_eotf(float3 val)
{
    const float3x3 agx_mat_inv = float3x3(
        1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116
    );

    // inverse input transform
    val = mul(agx_mat_inv, val);

    // value is now in nonlinear space; if RT is linear and swapchain is SRGB, uncomment
    // TODO: use proper value after TAA experimentation
    // val = pow(val, float3(2.2));

    return val;
}

[shader("compute")]
[numthreads(32, 32, 1)]
void main(
    uint3 global_id : SV_DispatchThreadID
)
{
	int2 texel_coord = int2(global_id.xy);

	if (texel_coord.x < pc.screen_size.x && texel_coord.y < pc.screen_size.y)
	{
		float3 color = images_rgba32f[pc.src_id].Load(texel_coord).rgb;

		float exposure = 1.0;

		// automatic exposure, see eq. 112: https://google.github.io/filament/Filament.md.html#imagingpipeline/physicallybasedcamera/automaticexposure
		// also, https://bruop.github.io/exposure/
		if (pc.autoexpose == 1.0)
		{
			float Lavg = *pc.luminance_avg_buffer;
			float Lmax = 9.6 * Lavg;
			exposure = 1.0 / Lmax;
		}

		color.xyz *= exposure;

		switch (pc.tonemap_func)
		{
			case 0:
			    color.xyz = agx(color.xyz);
				color.xyz = agx_eotf(color.xyz);
				break;
			case 1: color.xyz = linear_to_srgb(reinhard(color.xyz)); break;
			case 2: color.xyz = filmic_optimized(color.xyz); break;
			case 3: color.xyz = linear_to_srgb(color.xyz); break; // OETF -> clamp
			default: break; // clamp only
		}

		images_rgba32f[pc.dst_id].Store(texel_coord, float4(color, 1.0));
	}
}
