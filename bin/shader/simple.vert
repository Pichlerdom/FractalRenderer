#version 330

// in_Position was bound to attribute index 0 and in_Color was bound to attribute index 1

in vec2 in_Position;
in vec2 in_Texture_Pos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
 

out vec2 frag_Texture_Pos;

void main(void) {
	// Set the position to the one defined in our vertex array
	
	mat4 mvp = projection * view * model;
	gl_Position = mvp * vec4(in_Position, 0.0f, 1.0f);

	// Pass the color on to the fragment shader
	frag_Texture_Pos = in_Texture_Pos;
}
