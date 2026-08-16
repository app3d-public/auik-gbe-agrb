#version 460
#include <common/clip.glsl>

#define AUIK_HAS_BORDER_BIT  0x1u
#define AUIK_HAS_RADIUS_BIT  0x2u
#define AUIK_HAS_CHECKER_BIT 0x4u
#define AUIK_CHECKER_ROWS    2.0

layout(location = 0) in vec2 in_local_pos;
layout(location = 1) flat in vec2 in_size;
layout(location = 2) flat in vec4 in_background_color;
layout(location = 3) flat in vec4 in_border_color;
layout(location = 4) flat in float in_border_radius;
layout(location = 5) flat in float in_border_thickness;
layout(location = 6) flat in uint in_corner_mask;
layout(location = 7) flat in uint in_flags;
layout(location = 8) flat in uint in_clip_id;
layout(location = 9) in vec2 in_pixel_pos;

layout(location = 0) out vec4 out_color;

layout(std430, set = 0, binding = 1) readonly buffer ClipRectsBuffer { vec4 clip_rects[]; };

float get_corner_radius(vec2 p, float radius, uint corner_mask)
{
    // bit0: top-left, bit1: top-right, bit2: bottom-right, bit3: bottom-left
    vec2 s = step(0.0, p);
    vec4 corner_weights = vec4((1.0 - s.x) * (1.0 - s.y), // TL
                               s.x * (1.0 - s.y),         // TR
                               s.x * s.y,                 // BR
                               (1.0 - s.x) * s.y          // BL
    );
    const uvec4 shift = uvec4(0, 1, 2, 3);
    uvec4 bits = (uvec4(corner_mask) >> shift) & 1u;

    return radius * dot(corner_weights, vec4(bits));
}

float sd_rounded_rect(vec2 p, vec2 half_size, float radius)
{
    vec2 q = abs(p) - half_size + vec2(radius);
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

vec4 fill_checker(vec2 local_pos, vec2 size, vec4 background_color)
{
    float checker_size = max(2.0, size.y / AUIK_CHECKER_ROWS);
    const vec3 checker_light = vec3(0.85);
    const vec3 checker_dark = vec3(0.65);
    vec2 checker_pos = local_pos + 0.5 * size;
    ivec2 cell = ivec2(floor(checker_pos / checker_size));
    bool odd = ((cell.x + cell.y) & 1) != 0;
    vec3 checker_rgb = odd ? checker_dark : checker_light;
    return vec4(mix(checker_rgb, background_color.rgb, background_color.a), 1.0);
}

void main()
{
    if (is_clipped(in_pixel_pos, clip_rects[in_clip_id])) discard;

    vec2 half_size = 0.5 * in_size;
    bool has_border = (in_flags & AUIK_HAS_BORDER_BIT) != 0u;
    bool has_radius = (in_flags & AUIK_HAS_RADIUS_BIT) != 0u;
    bool has_checker = (in_flags & AUIK_HAS_CHECKER_BIT) != 0u;
    // Fast path: plain rect (no radius).
    if (!has_radius)
    {
        vec2 d = abs(in_local_pos) - half_size;
        if (d.x > 0.0 || d.y > 0.0) discard;

        vec4 fill_color = has_checker ? fill_checker(in_local_pos, in_size, in_background_color) : in_background_color;
        vec4 color = fill_color;
        if (has_border)
        {
            float thickness = max(in_border_thickness, 0.0);
            vec2 inner_half = max(half_size - vec2(thickness), vec2(0.0));
            bool inside_inner = abs(in_local_pos.x) <= inner_half.x && abs(in_local_pos.y) <= inner_half.y;
            color = inside_inner ? fill_color : in_border_color;
        }

        out_color = color;
        return;
    }

    // Allow stronger roundness than classic rounded-rect clamp.
    // Keep a finite upper bound for stability on tiny quads.
    float radius = clamp(in_border_radius, 0.0, max(half_size.x, half_size.y));
    float corner_radius = get_corner_radius(in_local_pos, radius, in_corner_mask);

    float dist_outer = sd_rounded_rect(in_local_pos, half_size, corner_radius);
    float aa_outer = max(0.5 * fwidth(dist_outer), 1e-4);
    float fill_outer = 1.0 - smoothstep(-aa_outer, aa_outer, dist_outer);

    if (fill_outer <= 0.0) discard;

    if (!has_border)
    {
        vec4 color = has_checker ? fill_checker(in_local_pos, in_size, in_background_color) : in_background_color;
        color.a *= fill_outer;
        out_color = color;
        return;
    }

    vec4 fill_color = has_checker ? fill_checker(in_local_pos, in_size, in_background_color) : in_background_color;
    vec4 color = fill_color;
    float thickness = max(in_border_thickness, 0.0);
    vec2 inner_half = max(half_size - vec2(thickness), vec2(0.0));
    float inner_radius = max(corner_radius - thickness, 0.0);
    float dist_inner = sd_rounded_rect(in_local_pos, inner_half, inner_radius);
    float aa_inner = max(fwidth(dist_inner), 1e-4);
    float fill_inner = 1.0 - smoothstep(0.0, aa_inner, dist_inner);

    float border_alpha = in_border_color.a * (1.0 - fill_inner);
    float fill_alpha = fill_color.a * fill_inner;
    float out_alpha = border_alpha + fill_alpha;
    if (out_alpha > 1e-5)
    {
        vec3 premul_rgb = in_border_color.rgb * border_alpha + fill_color.rgb * fill_alpha;
        color.rgb = premul_rgb / out_alpha;
        color.a = out_alpha;
    }
    else color = vec4(0.0);

    color.a *= fill_outer;
    out_color = color;
}
