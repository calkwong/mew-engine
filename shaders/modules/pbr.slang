// https://google.github.io/filament/Filament.md.html
// https://github.com/boksajak/brdf/blob/master/brdf.h

module pbr;

static const float PI = 3.14159265359;

public float D_GGX(float NdotH, float roughness)
{
	float a = NdotH * roughness;
	float k = roughness / (1.0 - NdotH * NdotH + a * a);
	return k * k * (1.0 / PI);
}

public float V_SmithGGXCorrelated(float NdotV, float NdotL, float roughness)
{
	float a2 = roughness * roughness;
	float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
	float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
	return 0.5 / (GGXV + GGXL);
}

public float3 F_Schlick(float u, float3 f0)
{
    float f = pow(1.0 - u, 5.0);
    return f + f0 * (1.0 - f);
}

public float F_Schlick(float u, float f0, float f90)
{
    return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
}

// see https://bruop.github.io/ibl/
public float3 F_SchlickRoughness(float u, float3 f0, float roughness)
{
	return f0 + (max(float3(1.0 - roughness), f0) - f0) * pow(1.0 - u, 5.0);
}

public float get_square_falloff_attenuation(float3 distance, float radius)
{
	float distance_square = dot(distance, distance);
	float light_inv_radius = 1.0 / radius;
	float factor = distance_square * light_inv_radius * light_inv_radius;
	float smooth_factor = max(1.0 - factor * factor, 0.0);
	return (smooth_factor * smooth_factor) / max(distance_square, 1e-4);
}

public float Fd_Lambert()
{
	return 1.0 / PI;
}

public float Fd_Burley(float NdotV, float NdotL, float LdotH, float roughness)
{
    float f90 = 0.5 + 2.0 * roughness * LdotH * LdotH;
    float light_scatter = F_Schlick(NdotL, 1.0, f90);
    float view_scatter = F_Schlick(NdotL, 1.0, f90);
    return light_scatter * view_scatter * (1.0 / PI);
}

public struct BRDFData
{
    public float3 N;
    public float3 L;
    public float3 V;
    public float3 H;
    public float3 f0;

    public float NdotL;
    public float NdotH;
    public float NdotV;
    public float VdotH;
    public float LdotH;
}

public BRDFData prepare_brdf_data(float3 N, float3 L, float3 V, float3 albedo, float roughness, float metallic)
{
    BRDFData data;

    data.N = N;
    data.L = L;
    data.V = V;

    float3 H = normalize(L + V);
    data.H = H;

    data.NdotL = max(dot(N, L), 0.0);
	data.NdotH = max(dot(N, H), 0.0);
	data.NdotV = max(dot(N, V), 0.001);
	data.VdotH = max(dot(V, H), 0.0);
	data.LdotH = max(dot(L, H), 0.0);

	// TODO: this should be a param
	float3 f0 = float3(0.04);
	data.f0 = lerp(f0, albedo, metallic);

	return data;
}

// for point lights only
public void update_brdf_data(inout BRDFData data, float3 L)
{
    data.L = L;
    data.H = normalize(data.L + data.V);;

    data.NdotL = max(dot(data.N, data.L), 0.0);
    data.NdotH = max(dot(data.N, data.H), 0.0);
    data.VdotH = max(dot(data.V, data.H), 0.0);
    data.LdotH = max(dot(data.L, data.H), 0.0);
}

public float3 eval_brdf(BRDFData brdf_data, float3 albedo, float roughness, float metallic)
{
	float D = D_GGX(brdf_data.NdotH, roughness);
	float G = V_SmithGGXCorrelated(brdf_data.NdotV, brdf_data.NdotL, roughness);
	float3 F = F_Schlick(brdf_data.VdotH, brdf_data.f0);

	float3 diffuse = Fd_Burley(brdf_data.NdotV, brdf_data.NdotL, brdf_data.LdotH, roughness) * albedo.xyz * (1.0 - metallic);
	float3 specular = D * G * F;

	// conflicting info on whether to modulate diffuse, leave it out unless we use Fd_Lambert
	return diffuse + specular;
}
