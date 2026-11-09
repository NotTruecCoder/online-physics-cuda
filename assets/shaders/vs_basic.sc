$input a_position, a_color0
$output v_color0, v_wpos

#include <bgfx_shader.sh>

// Low-poly coloured primitive (GDD §14.3): transform by the ortho/pseudo-iso view-projection and
// carry the per-vertex colour straight through. All scene geometry (disc/quad/cube) uses this.
void main()
{
	gl_Position = mul(u_modelViewProj, vec4(a_position, 1.0));
	v_color0 = a_color0;
	v_wpos = a_position;
}
