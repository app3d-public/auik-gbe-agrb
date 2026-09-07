#pragma once

#include <agrb/vector.hpp>
#include <auik/gbe/agrb/agrb.hpp>
#include <core/pipelines/stream_data.hpp>
#include <cstring>
#include "context.hpp"
#include "dirty_pages.hpp"

namespace auik::detail
{
    struct QuadsStyleData
    {
        u32 background_color = 0u;
        u32 border_color = 0u;
        f32 border_radius = 0.0f;
        f32 border_thickness = 0.0f;
        f32 z_order = 0.0f;
        u32 mask = 0u;
    };

    struct TexturesStyleData
    {
        amal::rect uv_rect;
        u32 tint_color = 0u;
        f32 z_order = 0.0f;
        u16 texture_id = 0u;
        u16 clip_id = 0u;
        u32 flags = 0u;
    };

    static_assert(sizeof(amal::rect) == 16u);
    static_assert(sizeof(QuadsStyleData) == 24u);
    static_assert(sizeof(TexturesStyleData) == 32u);

    template <typename InstanceData>
    struct InstanceSplitTraits;

    template <>
    struct InstanceSplitTraits<QuadsInstanceData>
    {
        using Style = QuadsStyleData;
        static amal::rect transform(const QuadsInstanceData &v) { return v.rect; }
        static Style style(const QuadsInstanceData &v)
        {
            return {v.background_color, v.border_color, v.border_radius, v.border_thickness, v.z_order, v.mask};
        }
        static bool equal(const Style &a, const Style &b)
        {
            return a.background_color == b.background_color && a.border_color == b.border_color &&
                   a.border_radius == b.border_radius && a.border_thickness == b.border_thickness &&
                   a.z_order == b.z_order && a.mask == b.mask;
        }
    };

    template <>
    struct InstanceSplitTraits<TexturesInstanceData>
    {
        using Style = TexturesStyleData;
        static amal::rect transform(const TexturesInstanceData &v) { return v.rect; }
        static Style style(const TexturesInstanceData &v)
        {
            return {v.uv_rect, v.tint_color, v.z_order, v.texture_id, v.clip_id, v.flags};
        }
        static bool equal(const Style &a, const Style &b)
        {
            return a.uv_rect.offset == b.uv_rect.offset && a.uv_rect.size == b.uv_rect.size &&
                   a.tint_color == b.tint_color && a.z_order == b.z_order && a.texture_id == b.texture_id &&
                   a.clip_id == b.clip_id && a.flags == b.flags;
        }
    };

    inline bool equal_instance_transform(const amal::rect &a, const amal::rect &b)
    {
        return a.offset == b.offset && a.size == b.size;
    }

    template <typename InstanceData>
    struct InstanceMasterData
    {
        // Canonical CPU state. Frame buffers are upload targets and are never read back for synchronization.
        using Style = typename InstanceSplitTraits<InstanceData>::Style;
        acul::vector<amal::rect> transforms;
        acul::vector<Style> styles;
        u32 transform_version = 0u;
        u32 style_version = 0u;
        DirtyPageState transform_pages;
        DirtyPageState style_pages;
    };

    template <typename InstanceData>
    struct InstanceStream
    {
        using InstanceType = InstanceData;
        using Style = typename InstanceSplitTraits<InstanceData>::Style;
        InstanceMasterData<InstanceData> *master = nullptr;
        agrb::vector<amal::rect> transforms;
        agrb::vector<Style> styles;
        vk::DescriptorSet descriptor_set;
        vk::Buffer descriptor_buffer_transforms = nullptr;
        vk::Buffer descriptor_buffer_styles = nullptr;
        vk::Buffer descriptor_buffer_clip_rects = nullptr;
        u32 transform_version = 0u;
        u32 style_version = 0u;
        DirtyPageState transform_pages;
        DirtyPageState style_pages;
        bool descriptor_buffer_transforms_dirty = true;
        bool descriptor_buffer_styles_dirty = true;
    };

    template <typename InstanceData>
    DrawDataID push_data_to_instance_stream(DrawStream *stream, const void *data, u32 frame_id)
    {
        using Traits = InstanceSplitTraits<InstanceData>;
        auto &gpu = static_cast<InstanceStream<InstanceData> *>(stream->stream_instances)[frame_id];
        auto &master = *gpu.master;
        const auto &value = *static_cast<const InstanceData *>(data);
        DrawDataID id{};
        id.render_id = static_cast<u32>(master.transforms.size());
        master.transforms.push_back(Traits::transform(value));
        master.styles.push_back(Traits::style(value));
        const u32 transform_version = next_cache_version(master.transform_version);
        const u32 style_version = next_cache_version(master.style_version);
        reset_dirty_pages(master.transform_pages, master.transforms.size(), transform_version);
        reset_dirty_pages(master.style_pages, master.styles.size(), style_version);
        const auto tr = gpu.transforms.push_back(master.transforms.back());
        const auto sr = gpu.styles.push_back(master.styles.back());
        if (tr & agrb::vector_result_flag_bits::buffer_reallocated) gpu.descriptor_buffer_transforms_dirty = true;
        if (sr & agrb::vector_result_flag_bits::buffer_reallocated) gpu.descriptor_buffer_styles_dirty = true;
        gpu.transform_version = transform_version;
        gpu.style_version = style_version;
        gpu.transform_pages = master.transform_pages;
        gpu.style_pages = master.style_pages;
        ++stream->draw_sizes[frame_id];
        return id;
    }

    template <typename InstanceData>
    void push_data_batch_to_instance_stream(DrawStream *stream, const void *data, u32 count, DrawDataID *out_ids,
                                            u32 frame_id)
    {
        if (count == 0u) return;
        using Traits = InstanceSplitTraits<InstanceData>;
        auto &gpu = static_cast<InstanceStream<InstanceData> *>(stream->stream_instances)[frame_id];
        auto &master = *gpu.master;
        const u32 base = static_cast<u32>(master.transforms.size());
        const auto *values = static_cast<const InstanceData *>(data);
        for (u32 i = 0u; i < count; ++i)
        {
            master.transforms.push_back(Traits::transform(values[i]));
            master.styles.push_back(Traits::style(values[i]));
            const auto tr = gpu.transforms.push_back(master.transforms.back());
            const auto sr = gpu.styles.push_back(master.styles.back());
            if (tr & agrb::vector_result_flag_bits::buffer_reallocated) gpu.descriptor_buffer_transforms_dirty = true;
            if (sr & agrb::vector_result_flag_bits::buffer_reallocated) gpu.descriptor_buffer_styles_dirty = true;
        }
        const u32 transform_version = next_cache_version(master.transform_version);
        const u32 style_version = next_cache_version(master.style_version);
        reset_dirty_pages(master.transform_pages, master.transforms.size(), transform_version);
        reset_dirty_pages(master.style_pages, master.styles.size(), style_version);
        gpu.transform_version = transform_version;
        gpu.style_version = style_version;
        gpu.transform_pages = master.transform_pages;
        gpu.style_pages = master.style_pages;
        stream->draw_sizes[frame_id] += count;
        if (!out_ids) return;
        for (u32 i = 0u; i < count; ++i)
        {
            out_ids[i].render_id = base + i;
            out_ids[i].hit_id = AUIK_INVALID_DRAW_DATA_ID;
        }
    }

    template <typename InstanceData>
    void update_split_instance(InstanceStream<InstanceData> &gpu, u32 id, const InstanceData &value)
    {
        using Traits = InstanceSplitTraits<InstanceData>;
        auto &master = *gpu.master;
        if (id >= master.transforms.size() || id >= gpu.transforms.size()) return;
        const auto transform = Traits::transform(value);
        const auto style = Traits::style(value);
        if (!equal_instance_transform(master.transforms[id], transform))
        {
            master.transforms[id] = transform;
            gpu.transforms[id] = transform;
            const u32 version = next_cache_version(master.transform_version);
            mark_dirty_page(master.transform_pages, master.transforms.size(), id, version);
            gpu.transform_version = version;
            mark_dirty_page(gpu.transform_pages, gpu.transforms.size(), id, version);
        }
        if (!Traits::equal(master.styles[id], style))
        {
            master.styles[id] = style;
            gpu.styles[id] = style;
            const u32 version = next_cache_version(master.style_version);
            mark_dirty_page(master.style_pages, master.styles.size(), id, version);
            gpu.style_version = version;
            mark_dirty_page(gpu.style_pages, gpu.styles.size(), id, version);
        }
    }

    template <typename InstanceData>
    void update_instance_stream_data(DrawStream *stream, DrawDataID id, const void *data, u32 frame_id)
    {
        auto &gpu = static_cast<InstanceStream<InstanceData> *>(stream->stream_instances)[frame_id];
        update_split_instance(gpu, id.render_id, *static_cast<const InstanceData *>(data));
    }

    template <typename InstanceData>
    void invalidate_instance_stream_data(DrawStream *stream, DrawDataID id, u32 frame_id)
    {
        auto &gpu = static_cast<InstanceStream<InstanceData> *>(stream->stream_instances)[frame_id];
        auto &master = *gpu.master;
        if (id.render_id >= master.transforms.size() || id.render_id >= gpu.transforms.size()) return;
        const amal::rect invalid{{-65536.0f, -65536.0f}, {0.0f, 0.0f}};
        if (equal_instance_transform(master.transforms[id.render_id], invalid)) return;
        master.transforms[id.render_id] = invalid;
        gpu.transforms[id.render_id] = invalid;
        const u32 version = next_cache_version(master.transform_version);
        mark_dirty_page(master.transform_pages, master.transforms.size(), id.render_id, version);
        gpu.transform_version = version;
        mark_dirty_page(gpu.transform_pages, gpu.transforms.size(), id.render_id, version);
    }

    template <typename InstanceData>
    void update_instance_stream_data_batch(DrawStream *stream, const DrawDataID *ids, const void *data, u32 count,
                                           u32 frame_id)
    {
        auto &gpu = static_cast<InstanceStream<InstanceData> *>(stream->stream_instances)[frame_id];
        const auto *values = static_cast<const InstanceData *>(data);
        for (u32 i = 0u; i < count; ++i) update_split_instance(gpu, ids[i].render_id, values[i]);
    }

    template <typename Stream>
    void clear_instance_stream(DrawStream *stream, u32 frame_id)
    {
        auto &gpu = static_cast<Stream *>(stream->stream_instances)[frame_id];
        auto &master = *gpu.master;
        if (!master.transforms.empty()) next_cache_version(master.transform_version);
        if (!master.styles.empty()) next_cache_version(master.style_version);
        master.transforms.clear();
        master.styles.clear();
        reset_dirty_pages(master.transform_pages, 0u, master.transform_version);
        reset_dirty_pages(master.style_pages, 0u, master.style_version);
        gpu.transforms.clear();
        gpu.styles.clear();
        gpu.transform_version = master.transform_version;
        gpu.style_version = master.style_version;
        gpu.transform_pages = master.transform_pages;
        gpu.style_pages = master.style_pages;
    }

    template <typename T>
    bool resize_instance_buffer(agrb::vector<T> &buffer, size_t count, bool &descriptor_dirty)
    {
        if (buffer.size() == count) return true;
        buffer.clear();
        const auto result = buffer.resize(count);
        if (result & agrb::vector_result_flag_bits::buffer_reallocated) descriptor_dirty = true;
        return result & agrb::vector_result_flag_bits::success;
    }

    template <typename Stream>
    void copy_instance_stream_frame_data(DrawStream *stream, u32 dst_id, u32 src_id)
    {
        if (dst_id == src_id) return;
        auto *frames = static_cast<Stream *>(stream->stream_instances);
        auto &dst = frames[dst_id];
        auto &master = *dst.master;
        const bool transform_size_changed = dst.transforms.size() != master.transforms.size();
        const bool style_size_changed = dst.styles.size() != master.styles.size();
        if (!resize_instance_buffer(dst.transforms, master.transforms.size(), dst.descriptor_buffer_transforms_dirty) ||
            !resize_instance_buffer(dst.styles, master.styles.size(), dst.descriptor_buffer_styles_dirty))
        {
            stream->draw_sizes[dst_id] = 0u;
            return;
        }
        if (transform_size_changed || dst.transform_version != master.transform_version)
            upload_dirty_pages(dst.transforms, master.transforms, dst.transform_pages, master.transform_pages,
                               transform_size_changed);
        if (style_size_changed || dst.style_version != master.style_version)
            upload_dirty_pages(dst.styles, master.styles, dst.style_pages, master.style_pages, style_size_changed);
        dst.transform_version = master.transform_version;
        dst.style_version = master.style_version;
        stream->draw_sizes[dst_id] = static_cast<u32>(master.transforms.size());
    }

    template <typename Stream>
    bool sync_instance_stream_cache(DrawStream *stream, void *sync_ctx, GPUContext *gpu_context, u32 frame_id)
    {
        (void)sync_ctx;
        (void)gpu_context;
        if (!stream->runtime_data) return false;
        auto *state = static_cast<StreamSyncState *>(stream->runtime_data);
        if (state->buffer_versions[frame_id] != state->master_version)
        {
            copy_instance_stream_frame_data<Stream>(stream, frame_id, state->master_id);
            state->buffer_versions[frame_id] = state->master_version;
            if (state->invalidation_count > 0u) --state->invalidation_count;
            if (state->invalidation_count == 0u) state->stage_version = state->master_version;
        }
        return state->invalidation_count > 0u;
    }

    template <typename Stream>
    void destroy_instance_stream_gpu_data(DrawStream *stream)
    {
        const u32 count = get_context().frames_in_flight;
        InstanceMasterData<typename Stream::InstanceType> *master = nullptr;
        for (u32 i = 0u; i < count; ++i)
        {
            auto &gpu = static_cast<Stream *>(stream->stream_instances)[i];
            if (!master) master = gpu.master;
            gpu.transforms.destroy();
            gpu.styles.destroy();
        }
        acul::release(master);
        acul::release(static_cast<Stream *>(stream->stream_instances), count);
    }

    template <typename Stream>
    void *create_instance_stream_gpu_data(u32 count, GPUContext *gpu_context)
    {
        auto *data = acul::alloc_n<Stream>(count);
        auto *master = acul::alloc<InstanceMasterData<typename Stream::InstanceType>>();
        auto &device = get_agrb_device(gpu_context);
        agrb::managed_buffer buffer{.required_flags = vk::MemoryPropertyFlagBits::eHostVisible |
                                                      vk::MemoryPropertyFlagBits::eHostCoherent,
                                    .buffer_usage = vk::BufferUsageFlagBits::eStorageBuffer,
                                    .vma_usage = VMA_MEMORY_USAGE_CPU_TO_GPU};
        for (u32 i = 0u; i < count; ++i)
        {
            data[i].master = master;
            data[i].transforms.init(device, buffer);
            data[i].styles.init(device, buffer);
        }
        return data;
    }

    template <typename Stream>
    bool update_instance_descriptor_set(DrawStream *stream, Stream &gpu, GPUContext *gpu_context, u32 frame_id)
    {
        auto *pipeline = stream->pipeline;
        auto *ctx = get_agrb_context(gpu_context);
        const vk::Buffer transforms = gpu.transforms.data().vk_buffer;
        const vk::Buffer styles = gpu.styles.data().vk_buffer;
        const vk::Buffer clips = ctx->clip_rects[frame_id].data().vk_buffer;
        const bool clips_changed = ctx->clip_rects_reallocated && ctx->clip_rects_reallocated[frame_id];
        if (!transforms || !styles || !clips) return false;
        if (gpu.descriptor_set && gpu.descriptor_buffer_transforms == transforms &&
            gpu.descriptor_buffer_styles == styles && gpu.descriptor_buffer_clip_rects == clips &&
            !gpu.descriptor_buffer_transforms_dirty && !gpu.descriptor_buffer_styles_dirty && !clips_changed)
            return true;
        vk::DescriptorBufferInfo transform_info{transforms, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo style_info{styles, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo clip_info{clips, 0, VK_WHOLE_SIZE};
        agrb::descriptor_writer writer(*pipeline->descriptor_set_layout, *ctx->descriptor_pool);
        writer.write_buffer(0, &transform_info);
        writer.write_buffer(1, &style_info);
        writer.write_buffer(2, &clip_info);
        if (!gpu.descriptor_set)
        {
            if (!writer.build(gpu.descriptor_set)) return false;
        }
        else writer.overwrite(gpu.descriptor_set);
        gpu.descriptor_buffer_transforms = transforms;
        gpu.descriptor_buffer_styles = styles;
        gpu.descriptor_buffer_clip_rects = clips;
        gpu.descriptor_buffer_transforms_dirty = false;
        gpu.descriptor_buffer_styles_dirty = false;
        return true;
    }
} // namespace auik::detail
