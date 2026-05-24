// based on https://github.com/KhronosGroup/glTF-Sample-Renderer/blob/15abf84b2ca03e138582958ec0876392cff0166a/source/Renderer/shaders/pbr.frag

module transmission;

import bindings;
import pbr;

public float3 eval_punctual_radiance_transmission(BRDFData brdf_data, float3 albedo, float roughness)
{
    // TODO: transmission roughness based on ior
    float transmission_roughness = roughness;

    // TODO: verify L direction
    float3 l_reflected = reflect(brdf_data.L, brdf_data.N);
    float3 H = normalize(l_reflected + brdf_data.V);

    float NdotH = max(dot(brdf_data.N, H), 0.0);
    float NdotLreflected = max(dot(brdf_data.N, l_reflected), 0.0);
    float D = D_GGX(NdotH, transmission_roughness);
    float V = V_SmithGGXCorrelated(NdotLreflected, brdf_data.NdotV, transmission_roughness);

    return albedo * D * V;
}

public float3 eval_ibl_volume_refraction(BRDFData brdf_data, float3 albedo, float perceptual_roughness, float3 pos, float4x4 view_proj, uint texture_index, float extent)
{
    // TODO: account for volume transmission ray

    float4 clip_pos = mul(view_proj, float4(pos, 1.0));
    float2 ndc = clip_pos.xy / clip_pos.w;
    float2 sample_uv = ndc * 0.5 + 0.5;
    sample_uv.y = 1.0 - sample_uv.y;

    // TODO: sample LOD based on roughness and ior
    float framebuffer_lod = log2(extent) * apply_ior_to_roughness(perceptual_roughness);
    framebuffer_lod = 0.0;

    // sample framebuffer
    float3 transmitted_light = textures[texture_index].SampleLevel(samplers[NEAREST_SAMPLER], sample_uv, framebuffer_lod).rgb;

    // TODO: account for volume attenuation
    float3 attenuated_color = transmitted_light;

    return attenuated_color * albedo;
}

public float apply_ior_to_roughness(float roughness, float ior = 1.5)
{
    // Scale roughness with IOR so that an IOR of 1.0 results in no microfacet refraction and
    // an IOR of 1.5 results in the default amount of microfacet refraction.
    return roughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
}
