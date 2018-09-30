#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec2 o_color;
layout (location = 0) out vec4 uFragColor;

layout (binding = 0) uniform Foo {
    bool b;
    uint i;
} foo;
void main() {
    uint i = 42;
    bool t = true;
    uFragColor = vec4(o_color, 0, 1);
}
