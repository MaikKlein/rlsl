#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 fragColor;

void main() {
    vec3 color = vec3(1, 0, 0);
    uint discr = uint(uv.x > 0.5);
    switch(discr) {
        case 1: color.x = 0.5; break;
        default: break;
    }
    fragColor = vec4(color, 1);
}
