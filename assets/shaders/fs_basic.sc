$input v_color0, v_wpos

#include <bgfx_shader.sh>

// What stands between you and the view steps aside: an oblong on the screen, from just in front of you toward the camera, above your
// feet. Toward its ends it goes into what is around it slowly. Under a roof, nothing over `ceiling` is drawn either. It is drawn twice:
// once for what is whole, writing depth, and once more for what is fading, blended over that (u_cutKeep.x says which). Nothing is moved.
uniform vec4 u_cut;     // xyz: where the tube starts, world; w: how wide it is, 0 for none
uniform vec4 u_cutDir;  // xyz: toward the camera; w: the height nothing below is touched
uniform vec4 u_cutKeep; // x: how much of what is in the tube is still drawn; y: the height over which nothing near you is; z: how wide the tube for trees is; w: how much of them stays
uniform vec4 u_clip;    // xy: one corner of a box on the ground, zw: the other; a coarse region is not drawn inside it (the live window is, at full detail). z <= x: none

// How much of it there is at this place in the oblong: 0 in the middle, 1 outside; the long way it fades over half its length.
float oblong(vec3 d, vec3 toward, float size)
{
	vec3 side = normalize(cross(toward, vec3(0.0, 0.0, 1.0)));
	vec3 upward = cross(side, toward);
	float wide = abs(dot(d, side)) / (size * 1.5);
	float tall = abs(dot(d, upward)) / (size * 0.85);
	return max(smoothstep(0.45, 1.0, wide), smoothstep(0.75, 1.0, tall));
}

void main()
{
	if (u_clip.z > u_clip.x && v_wpos.x > u_clip.x && v_wpos.x < u_clip.z && v_wpos.y > u_clip.y && v_wpos.y < u_clip.w) {
		discard;
	}
	float seen = 1.0;
	if ((u_cut.w > 0.0 || u_cutKeep.z > 0.0) && v_wpos.z > u_cutDir.w) {
		vec3 d = v_wpos - u_cut.xyz;
		if (dot(d, u_cutDir.xyz) > 0.0) {
			if (u_cut.w > 0.0) {
				seen = oblong(d, u_cutDir.xyz, u_cut.w);
			}
			// A tree says so in the alpha of its colour (VoxelMesh.cpp): nothing else is that one value.
			if (u_cutKeep.z > 0.0 && abs(v_color0.a - 254.0 / 255.0) < 0.002) {
				seen = min(seen, oblong(d, u_cutDir.xyz, u_cutKeep.z));
			}
		}
		if (u_cut.w > 0.0 && u_cutKeep.y > 0.0 && v_wpos.z > u_cutKeep.y) {
			seen = min(seen, smoothstep(0.75, 1.0, length(d.xy) / u_cut.w));
		}
	}
	vec4 colour = v_color0;
	if (u_cutKeep.x > 0.5) {
		if (seen >= 0.999 || seen <= 0.01) {
			discard;
		}
		colour.a = seen;
	} else if (seen < 0.999) {
		discard;
	}
	gl_FragColor = colour;
}
