$input a_position, a_texcoord0
$output v_texcoord0

#include <bgfx_shader.sh>

// Fullscreen pass: positions are already in clip space; just forward UV.
void main()
{
	gl_Position = vec4(a_position.xy, 0.0, 1.0);
	v_texcoord0 = a_texcoord0;
}
