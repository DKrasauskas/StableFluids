#version 460 core
layout(location = 0) in vec3 aPos;
out vec4 Color;
layout(location = 2) uniform float max_p = 0;
layout(location = 3) uniform float min_p = 0;

float lininterp(float value) {
	return (value - min_p) / (max_p - min_p) * 1.0f;
}
layout(std430, binding = 1) buffer colors
{
	float data[];
};
layout(std430, binding = 2) buffer colors1
{
	float data1[];
};
layout(std430, binding = 3) buffer colors2
{
	float data2[];
};

vec4 color_map(float data) {
	if (data < 0.166f) {
		return vec4(data * 6, 0.0, 0.0, 1.0f);
	}
	if (data < 0.333f) {
		return vec4(1.0f, (data - 0.1666f) * 6, 0.0, 1.0);
	}
	if (data < 0.5f) {
		return vec4(1.0 - (data - 0.33f) * 6, 1.0, 0.0, 1.0);
	}
	if (data < 0.666f) {
		return vec4(0.0, 1.0, (data - 0.5f) * 6, 1.0);
	}
	if (data < 0.8333f) {
		return vec4(0.0, 1.0 - (data - 0.66f) * 6, 1.0, 1.0);
	}
	return vec4(0, 0, 1, 1);
}
void main() {
	gl_Position = vec4(aPos.x * 2, aPos.y * 2, aPos.z, 1.0);
	Color = color_map(lininterp(data[gl_VertexID]));
}
