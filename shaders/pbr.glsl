// Based on https://google.github.io/filament/Filament.md.html

#ifndef PI
#define PI 3.14159265359
#endif

float D_GGX(float NdotH, float roughness)
{
	float a = NdotH * roughness;
	float k = roughness / (1.0 - NdotH * NdotH + a * a);
	return k * k * (1.0 / PI);
}

float V_SmithGGXCorrelated(float NdotV, float NdotL, float roughness)
{
	float a2 = roughness * roughness;
	float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
	float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
	return 0.5 / (GGXV + GGXL);
}

vec3 F_Schlick(float u, vec3 f0) 
{
    float f = pow(1.0 - u, 5.0);
    return f + f0 * (1.0 - f);
}

// takes NdotV, not VdotH
vec3 F_SchlickRoughness(float u, vec3 f0, float roughness)
{
	return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(1.0 - u, 5.0);
}

float get_square_falloff_attenuation(vec3 distance, float radius)
{
	float distance_square = dot(distance, distance);
	float light_inv_radius = 1.0 / radius;
	float factor = distance_square * light_inv_radius * light_inv_radius;
	float smooth_factor = max(1.0 - factor * factor, 0.0);
	return (smooth_factor * smooth_factor) / max(distance_square, 1e-4);
}