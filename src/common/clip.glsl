bool is_clipped(vec2 pixel_pos, vec4 clip_rect)
{
    vec2 clip_min = clip_rect.xy;
    vec2 clip_max = clip_rect.xy + clip_rect.zw;
    return pixel_pos.x < clip_min.x || pixel_pos.y < clip_min.y || pixel_pos.x >= clip_max.x ||
           pixel_pos.y >= clip_max.y;
}
