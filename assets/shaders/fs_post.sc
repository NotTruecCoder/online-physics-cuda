$input v_texcoord0

#include <bgfx_shader.sh>

SAMPLER2D(s_tex, 0);

// u_post: x = pixel_size (unused here; pixelization comes from the low-res source), y = colour
// levels for palette quantization, z/w reserved.
uniform vec4 u_post;

void main()
{
	// The source is a low-resolution render target sampled with point filtering, so upscaling it to
	// the backbuffer already yields blocky pixels. Posterize the colour to the limited palette.
	vec4 c = texture2D(s_tex, v_texcoord0);
	float levels = max(u_post.y, 2.0);
	c.rgb = floor(c.rgb * levels + 0.5) / levels;
	gl_FragColor = vec4(c.rgb, 1.0);
}
