#version 330
// It was expressed that some drivers required this next line to function properly
precision highp float;
 
in vec2 frag_Texture_Pos;

out vec4 fragColor;

uniform sampler2D sampler;

void main(void) {
  fragColor = vec4(texture(sampler, frag_Texture_Pos).rgb,1.0f);
}

