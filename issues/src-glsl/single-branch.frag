#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 fragColor;

void main() {
    vec3 color = vec3(1, 0, 0);
    if(uv.x > 0.5) {
        color.x = 0.5;
    }
    fragColor = vec4(color, 1);
}
