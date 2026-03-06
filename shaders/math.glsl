#ifndef MATH_GLSL
#define MATH_GLSL

#define PI 3.14159265359

// http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 tonemap(vec3 color)
{
   vec3 x = max(vec3(0.0), color - 0.004);
   vec3 retColor = (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
   
   return retColor;
}

#endif