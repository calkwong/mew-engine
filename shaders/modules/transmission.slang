// based on https://github.com/KhronosGroup/glTF-Sample-Renderer/blob/15abf84b2ca03e138582958ec0876392cff0166a/source/Renderer/shaders/pbr.frag
// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md
// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_volume/README.md

module transmission;

import bindings;
import pbr;

public float3 eval_punctual_radiance_transmission(BRDFData brdf_data, float3 L, float3 albedo, float roughness, float ior)
{
    float transmission_roughness = apply_ior_to_roughness(roughness, ior);

    // TODO: verify L direction
    float3 l_reflected = reflect(L, brdf_data.N);
    float3 H = normalize(l_reflected + brdf_data.V);

    float NdotH = max(dot(brdf_data.N, H), 0.0);
    float NdotLreflected = max(dot(brdf_data.N, l_reflected), 0.0);
    float D = D_GGX(NdotH, transmission_roughness);
    float V = V_SmithGGXCorrelated(NdotLreflected, brdf_data.NdotV, transmission_roughness);

    return albedo * D * V;
}

public float3 get_transmission_sample(float2 uv, uint texture_index, float extent, float perceptual_roughness, float ior)
{
    float framebuffer_lod = log2(extent) * apply_ior_to_roughness(perceptual_roughness, ior);
    float3 transmitted_light = textures[texture_index].SampleLevel(samplers[NEAREST_SAMPLER], uv, framebuffer_lod).rgb;

    return transmitted_light;
}

public float3 eval_ibl_volume_refraction(BRDFData brdf_data, float3 albedo, float perceptual_roughness, float3 pos, float4x4 view_proj, uint texture_index, float extent,
    float thickness, float3 attenuation_color, float attenuation_distance, float ior, float scale
)
{
    float3 transmission_ray = get_volume_transmission_ray(brdf_data, thickness, ior, scale);
    float transmission_ray_length = length(transmission_ray);
    float3 refracted_ray_exit = pos + transmission_ray;

    float4 clip_pos = mul(view_proj, float4(refracted_ray_exit, 1.0));
    float2 ndc = clip_pos.xy / clip_pos.w;
    float2 sample_uv = ndc * 0.5 + 0.5;
    sample_uv.y = 1.0 - sample_uv.y;

    float3 transmitted_light = get_transmission_sample(sample_uv, texture_index, extent, perceptual_roughness, ior);

    // noop if volume is off
    float3 attenuated_color = apply_volume_attenuation(transmitted_light, transmission_ray_length, attenuation_color, attenuation_distance);

    return attenuated_color * albedo;
}

// Scale roughness with IOR so that an IOR of 1.0 results in no microfacet refraction and
// an IOR of 1.5 results in the default amount of microfacet refraction.
public float apply_ior_to_roughness(float roughness, float ior)
{
    return roughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
}

public float3 apply_volume_attenuation(float3 radiance, float transmission_distance, float3 attenuation_color, float attenuation_distance)
{
    // attenuation_distance of 0.0 represents infinity, which means transmitted color is not attenuated
    if (attenuation_distance == 0.0)
    {
        return radiance;
    }
    else // apply beer's law
    {
        float3 transmittance = pow(attenuation_color, float3(transmission_distance / attenuation_distance));
        return transmittance * radiance;
    }
}

public float3 get_volume_transmission_ray(BRDFData brdf_data, float thickness, float ior, float scale)
{
    // direction of refracted light
    float3 refracted_vector = refract(-brdf_data.V, brdf_data.N, 1.0 / ior);

    // compute rotation-independent scaling of model matrix
    // float3 model_scale;
    // model_scale.x = length(model_matrix[0].xyz);
    // model_scale.y = length(model_matrix[1].xyz);
    // model_scale.z = length(model_matrix[2].xyz);

    // thickness specified in local space
    // note: we are making an assumption that scaling is uniform here
    // this will break under non uniform scaling
    return normalize(refracted_vector) * thickness * scale; //model_scale;
}
