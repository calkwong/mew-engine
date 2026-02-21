// From http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/

struct BarycentricDeriv
{
	vec3 lambda;
	vec3 ddx;
	vec3 ddy;
};

BarycentricDeriv calculateBarycentric(vec4 p0, vec4 p1, vec4 p2, vec2 pNdc, vec2 winSize)
{
	BarycentricDeriv ret;
	
	vec3 invW = 1.0 / vec3(p0.w, p1.w, p2.w);
	
	// perspective divide
	vec2 ndc0 = p0.xy * invW.x;
	vec2 ndc1 = p1.xy * invW.y;
	vec2 ndc2 = p2.xy * invW.z;
	
	// HLSL is row major, GLSL is column major. Determinant M == Determinant transposeM.
	float invDet = 1.0 / determinant(mat2(ndc2 - ndc1, ndc0 - ndc1));
	// change in barycentric coords given a unit change in x/y in ndc space, perspective corrected
	// review formula, why do we multiply with invDet again? 
	ret.ddx = vec3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * invDet * invW;
	ret.ddy = vec3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * invDet * invW;
	// denominator for perspective correct barycentrics
	float ddxSum = ret.ddx.x + ret.ddx.y + ret.ddx.z;
	float ddySum = ret.ddy.x + ret.ddy.y + ret.ddy.z;
	
	vec2 deltaVec = pNdc - ndc0;
	float interpInvW = invW.x + deltaVec.x * ddxSum + deltaVec.y * ddySum;
	float interpW = 1.0 / interpInvW;
	
	ret.lambda.x = interpW * (invW[0] + deltaVec.x * ret.ddx.x + deltaVec.y * ret.ddy.x);
	ret.lambda.y = interpW * (0.0     + deltaVec.x * ret.ddx.y + deltaVec.y * ret.ddy.y);
	ret.lambda.z = interpW * (0.0     + deltaVec.x * ret.ddx.z + deltaVec.y * ret.ddy.z);
	
	// ndc to pixel 
	ret.ddx *= (2.0 / winSize.x);
	ret.ddy *= (2.0 / winSize.y);
	ddxSum *= (2.0 / winSize.x);
	ddySum *= (2.0 / winSize.y);
	
	// flip as ndc is bottom-top while window coords are top down
	ret.ddy *= -1.0;
	ddySum *= -1.0;
	
	float interpW_ddx = 1.0 / (interpInvW + ddxSum);
	float interpW_ddy = 1.0 / (interpInvW + ddySum);
	
	ret.ddx = interpW_ddx * (ret.lambda * interpInvW + ret.ddx) - ret.lambda;
	ret.ddy = interpW_ddy * (ret.lambda * interpInvW + ret.ddy) - ret.lambda;
	
	return ret;
}

vec4 interpolate(BarycentricDeriv bary, vec4 v0, vec4 v1, vec4 v2)
{
	return bary.lambda.x * v0 + bary.lambda.y * v1 + bary.lambda.z * v2;
}

vec3 interpolate(BarycentricDeriv bary, vec3 v0, vec3 v1, vec3 v2)
{
	return bary.lambda.x * v0 + bary.lambda.y * v1 + bary.lambda.z * v2;
}

vec2 interpolate(BarycentricDeriv bary, vec2 v0, vec2 v1, vec2 v2)
{
	return bary.lambda.x * v0 + bary.lambda.y * v1 + bary.lambda.z * v2;
}

float interpolate(BarycentricDeriv bary, float v0, float v1, float v2)
{
	return bary.lambda.x * v0 + bary.lambda.y * v1 + bary.lambda.z * v2;
}

void interpolateWithDeriv(BarycentricDeriv bary, vec2 v0, vec2 v1, vec2 v2, inout vec2 v, inout vec2 ddx, inout vec2 ddy)
{
	v = bary.lambda.x * v0 + bary.lambda.y * v1 + bary.lambda.z * v2;
	ddx = bary.ddx.x * v0 + bary.ddx.y * v1 + bary.ddx.z * v2;
	ddy = bary.ddy.x * v0 + bary.ddy.y * v1 + bary.ddy.z * v2;
}