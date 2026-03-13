#ifndef PI
#define PI 3.14159265359
#endif

SH9 shMul(SH9 a, SH9 b)
{
	for (int i = 0; i < 9; i++)
	{
		a.c[i] = a.c[i] * b.c[i];
	}
	
	return a;
}

SH9 shSum(SH9 a, SH9 b)
{
	for (int i = 0; i < 9; i++)
	{
		a.c[i] = a.c[i] + b.c[i];
	}
	
	return a;
}

SH9 shScale(SH9 a, float value)
{
	for (int i = 0; i < 9; i++)
	{
		a.c[i] *= value;
	}
	
	return a;
}

float shDot(SH9 a, SH9 b)
{
	float value = 0.0;

	for (int i = 0; i < 9; i++)
	{
		value += a.c[i] * b.c[i];
	}
	
	return value;
}

SH9 shBasis(vec3 dir)
{
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;
	
	SH9 base;
	base.c[0] = 1;           // band 0
	base.c[1] = y;           // band 1
	base.c[2] = z;
	base.c[3] = x;
	base.c[4] = x * y;       // band 2
	base.c[5] = y * z;
	base.c[6] = 3 * z * z - 1;
	base.c[7] = x * z;
	base.c[8] = x * x - y * y;
	
	return base;
}

SH9 shZero()
{
	SH9 sh;
	for (int i = 0; i < 9; i++)
	{
		sh.c[i] = 0.0;
	}
	
	return sh;
}

// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#fragment-SamplingFunctionDefinitions-6
vec3 uniformSampleSphere(float az, float ze)
{
	float phi = 2.0 * PI * az;
	float z = 1.0 - 2.0 * ze; // mapping from hemisphere -> sphere
	float r = sqrt(max(0.0, 1.0 - z * z));
	
	return vec3(r * cos(phi), z, r * sin(phi));
}

vec3 evaluateSH(SH9 rCoefficients, SH9 gCoefficients, SH9 bCoefficients, vec3 direction)
{
	SH9 coefficients;
	coefficients.c[0] = 0.282095; 
	coefficients.c[1] = -0.488603;
	coefficients.c[2] = 0.488603;
	coefficients.c[3] = -0.488603;
	coefficients.c[4] = 1.092548;
	coefficients.c[5] = -1.092548;
	coefficients.c[6] = 0.315392;
	coefficients.c[7] = -1.092548;
	coefficients.c[8] = 0.546274;

	SH9 base = shBasis(direction);
	
	base = shMul(coefficients, base);
	
	float r = max(shDot(rCoefficients, base), 0.0);
	float g = max(shDot(gCoefficients, base), 0.0);
	float b = max(shDot(bCoefficients, base), 0.0);
	
	return vec3(r, g, b);
}