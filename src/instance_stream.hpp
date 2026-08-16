#pragma once

#include <agrb/vector.hpp>
#include <auik/gbe/agrb/agrb.hpp>
#include <core/pipelines/stream_data.hpp>
#include "context.hpp"

namespace auik::detail
{
    inline void invalidate_instance_payload(QuadsInstanceData &data)
    {
        data.rect = {{-65536.0f, -65536.0f}, {0.0f, 0.0f}};
    }

    inline void invalidate_instance_payload(TexturesInstanceData &data)
    {
        data.rect = {{-65536.0f, -65536.0f}, {0.0f, 0.0f}};
    }

    template <typename InstanceData>
    struct InstanceStream
    {
        agrb::vector<InstanceData> draw_instances;
        vk::DescriptorSet descriptor_set;
        vk::Buffer descriptor_buffer_instances = nullptr;
        vk::Buffer descriptor_buffer_clip_rects = nullptr;
        bool descriptor_buffer_instances_dirty = true;
    };

    template <typename InstanceData>
    DrawDataID push_data_to_instance_stream(DrawStream *stream, const void *data, u32 frame_id)
    {
        auto &gpu_data = static_cast<InstanceStream<InstanceData> *>(stream->stream_instances)[frame_id];
        DrawDataID draw_data_id{};
        draw_data_id.render_id = static_cast<u32>(gpu_data.draw_instances.size());
        const auto result = gpu_data.draw_instances.push_back(*static_cast<const InstanceData *>(data));
        if (result & agrb::VectorResultBits::buffer_reallocated) gpu_data.descriptor_buffer_instances_dirty = true;
        ++stream->draw_sizes[frame_id];
        return draw_data_id;
    }

    template <typename InstanceData>
    void push_data_batch_to_instance_stream(DrawStream *stream, const void *data, u32 count, DrawDataID *out_draw_ids,
                                            u32 frame_id)
    {
        if (count == 0) return;

        auto &gpu_data = static_cast<InstanceStream<InstanceData> *>(stream->stream_instances)[frame_id];
        const u32 base_render_id = static_cast<u32>(gpu_data.draw_instances.size());
        const auto *instances = static_cast<const InstanceData *>(data);
        for (u32 i = 0; i < count; ++i)
        {
            const auto result = gpu_data.draw_instances.push_back(instances[i]);
            if (result & agrb::VectorResultBits::buffer_reallocated) gpu_data.descriptor_buffer_instances_dirty = true;
        }
        stream->draw_sizes[frame_id] += count;

        if (!out_draw_ids) return;
        for (u32 i = 0; i < count; ++i)
        {
            out_draw_ids[i].render_id = base_render_id + i;
            out_draw_ids[i].hit_id = AUIK_INVALID_DRAW_DATA_ID;
        }
    }

    template <typename InstanceData>
    void update_instance_stream_data(DrawStream *stream, DrawDataID draw_data_id, const void *data, u32 frame_id)
    {
        auto &gpu_data = static_cast<InstanceStream<InstanceData> *>(stream->stream_instances)[frame_id];
        if (draw_data_id.render_id >= gpu_data.draw_instances.size()) return;
        gpu_data.draw_instances[draw_data_id.render_id] = *static_cast<const InstanceData *>(data);
    }

    template <typename InstanceData>
    void invalidate_instance_stream_data(DrawStream *stream, DrawDataID draw_data_id, u32 frame_id)
    {
        auto &gpu_data = static_cast<InstanceStream<InstanceData> *>(stream->stream_instances)[frame_id];
        if (draw_data_id.render_id >= gpu_data.draw_instances.size()) return;
        invalidate_instance_payload(gpu_data.draw_instances[draw_data_id.render_id]);
    }

    template <typename InstanceData>
    void update_instance_stream_data_batch(DrawStream *stream, const DrawDataID *draw_data_ids, const void *data,
                                           u32 count, u32 frame_id)
    {
        if (count == 0) return;

        auto &gpu_data = static_cast<InstanceStream<InstanceData> *>(stream->stream_instances)[frame_id];
        const auto *instances = static_cast<const InstanceData *>(data);
        for (u32 i = 0; i < count; ++i)
        {
            const u32 render_id = draw_data_ids[i].render_id;
            if (render_id >= gpu_data.draw_instances.size()) continue;
            gpu_data.draw_instances[render_id] = instances[i];
        }
    }

    template <typename Stream>
    void clear_instance_stream(DrawStream *stream, u32 frame_id)
    {
        auto &gpu_data = static_cast<Stream *>(stream->stream_instances)[frame_id];
        gpu_data.draw_instances.clear();
    }

    template <typename Stream>
    void copy_instance_stream_frame_data(DrawStream *stream, u32 dst_frame_id, u32 src_frame_id)
    {
        if (dst_frame_id == src_frame_id) return;
        auto *frames = static_cast<Stream *>(stream->stream_instances);
        auto &dst = frames[dst_frame_id];
        auto &src = frames[src_frame_id];
        dst.draw_instances.clear();
        for (u32 i = 0; i < src.draw_instances.size(); ++i)
        {
            const auto result = dst.draw_instances.push_back(src.draw_instances[i]);
            if (result & agrb::VectorResultBits::buffer_reallocated) dst.descriptor_buffer_instances_dirty = true;
        }
        stream->draw_sizes[dst_frame_id] = stream->draw_sizes[src_frame_id];
    }

    template <typename Stream>
    bool sync_instance_stream_cache(DrawStream *stream, void *sync_ctx, GPUContext *gpu_context, u32 frame_id)
    {
        (void)sync_ctx;
        if (!stream->runtime_data) return false;

        auto *state = static_cast<StreamSyncState *>(stream->runtime_data);
        assert(state && state->buffer_versions);
        if (state->buffer_versions[frame_id] != state->master_version)
        {
            copy_instance_stream_frame_data<Stream>(stream, frame_id, state->master_id);
            state->buffer_versions[frame_id] = state->master_version;
            if (state->invalidation_count > 0) --state->invalidation_count;
            if (state->invalidation_count == 0) state->stage_version = state->master_version;
        }

        return state->invalidation_count > 0;
    }

    template <typename Stream>
    void destroy_instance_stream_gpu_data(DrawStream *stream)
    {
        u32 count = get_context().frames_in_flight;
        for (u32 i = 0; i < count; i++)
        {
            auto &gpu_data = static_cast<Stream *>(stream->stream_instances)[i];
            gpu_data.draw_instances.destroy();
        }
        acul::release(static_cast<Stream *>(stream->stream_instances), count);
    }

    template <typename Stream>
    void *create_instance_stream_gpu_data(u32 instance_count, GPUContext *gpu_context)
    {
        auto *data = acul::alloc_n<Stream>(instance_count);
        auto &device = get_agrb_device(gpu_context);
        agrb::managed_buffer buf{.required_flags = vk::MemoryPropertyFlagBits::eHostVisible |
                                                   vk::MemoryPropertyFlagBits::eHostCoherent,
                                 .buffer_usage = vk::BufferUsageFlagBits::eStorageBuffer,
                                 .vma_usage = VMA_MEMORY_USAGE_CPU_TO_GPU};
        for (u32 i = 0; i < instance_count; i++) data[i].draw_instances.init(device, buf);
        return data;
    }

    template <typename Stream>
    bool update_instance_descriptor_set(DrawStream *stream, Stream &gpu_data, GPUContext *gpu_context, u32 frame_id)
    {
        auto *pipeline = stream->pipeline;
        auto *ctx = get_agrb_context(gpu_context);
        assert(pipeline && pipeline->descriptor_set_layout);
        assert(ctx->clip_rects);

        const auto &instances_data = gpu_data.draw_instances.data();
        const auto &clip_rects_data = ctx->clip_rects[frame_id].data();
        const vk::Buffer instance_buffer = instances_data.vk_buffer;
        const vk::Buffer clip_rects_buffer = clip_rects_data.vk_buffer;
        const bool clip_rects_reallocated = ctx->clip_rects_reallocated && ctx->clip_rects_reallocated[frame_id];
        if (!instance_buffer || !clip_rects_buffer) return false;

        if (gpu_data.descriptor_set && gpu_data.descriptor_buffer_instances == instance_buffer &&
            gpu_data.descriptor_buffer_clip_rects == clip_rects_buffer && !gpu_data.descriptor_buffer_instances_dirty &&
            !clip_rects_reallocated)
            return true;

        vk::DescriptorBufferInfo instance_info{instance_buffer, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo clip_rects_info{clip_rects_buffer, 0, VK_WHOLE_SIZE};
        agrb::descriptor_writer writer(*pipeline->descriptor_set_layout, *ctx->descriptor_pool);
        writer.write_buffer(0, &instance_info);
        writer.write_buffer(1, &clip_rects_info);
        if (!gpu_data.descriptor_set)
        {
            if (!writer.build(gpu_data.descriptor_set)) return false;
        }
        else writer.overwrite(gpu_data.descriptor_set);

        gpu_data.descriptor_buffer_instances = instance_buffer;
        gpu_data.descriptor_buffer_clip_rects = clip_rects_buffer;
        gpu_data.descriptor_buffer_instances_dirty = false;
        return true;
    }
} // namespace auik::detail
