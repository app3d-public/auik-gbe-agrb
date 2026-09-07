#include <acul/scalars.hpp>
#include <acul/string/string.hpp>
#include <acul/vector.hpp>
#include <auik-gbe-agrb/shaders.h>
#include <auik/gbe/agrb/textured_vertex_pipeline.hpp>
#include <auik/pipelines.hpp>
#include <cstddef>
#include <cstring>
#include "../context.hpp"
#include "../dirty_pages.hpp"

namespace auik::detail
{
    struct TexturedVertexStreamBatchRange
    {
        u32 vertex_offset = 0;
        u32 vertex_count = 0;
        u32 index_offset = 0;
        u32 index_count = 0;
    };

    struct TexturedVertexBatchStyle
    {
        u32 texture_id = AUIK_INVALID_DRAW_DATA_ID;
        u32 flags = 0u;
    };

    static_assert(sizeof(TexturedVertexBatchStyle) == 8u);

    struct TexturedVertexStreamMasterData
    {
        acul::vector<TexturedVertexStreamVertex> vertices;
        acul::vector<TexturedVertexStreamIndex> indices;
        acul::vector<amal::vec2> offsets;
        acul::vector<TexturedVertexBatchStyle> styles;
        acul::vector<TexturedVertexStreamBatchRange> batches;
        u32 vertices_version = 0u;
        u32 indices_version = 0u;
        u32 offsets_version = 0u;
        u32 styles_version = 0u;
        DirtyPageState vertices_pages;
        DirtyPageState indices_pages;
        DirtyPageState offsets_pages;
        DirtyPageState styles_pages;
    };

    struct TexturedVertexStreamGPUData
    {
        TexturedVertexStreamMasterData *master = nullptr;
        agrb::vector<TexturedVertexStreamVertex> vertices;
        agrb::vector<TexturedVertexStreamIndex> indices;
        agrb::vector<amal::vec2> offsets;
        agrb::vector<TexturedVertexBatchStyle> styles;
        vk::DescriptorSet descriptor_set;
        vk::Buffer descriptor_buffer_clip_rects = nullptr;
        vk::Buffer descriptor_buffer_offsets = nullptr;
        vk::Buffer descriptor_buffer_styles = nullptr;
        u32 vertices_version = 0u;
        u32 indices_version = 0u;
        u32 offsets_version = 0u;
        u32 styles_version = 0u;
        DirtyPageState vertices_pages;
        DirtyPageState indices_pages;
        DirtyPageState offsets_pages;
        DirtyPageState styles_pages;
        bool descriptor_buffer_offsets_dirty = true;
        bool descriptor_buffer_styles_dirty = true;
    };

    static DrawDataID push_textured_vertex_stream_batch(DrawStream *stream, const TexturedVertexStreamBatchData &batch,
                                                        u32 frame_id)
    {
        DrawDataID draw_data_id{};
        if ((!batch.vertices && batch.vertex_count > 0) || (!batch.indices && batch.index_count > 0))
            return draw_data_id;
        if (batch.texture_id.handle == 0 || batch.texture_id.bind_slot == AUIK_INVALID_DRAW_DATA_ID)
            return draw_data_id;

        auto &gpu_data = static_cast<TexturedVertexStreamGPUData *>(stream->stream_instances)[frame_id];
        auto &master = *gpu_data.master;
        const u32 batch_id = static_cast<u32>(master.batches.size());
        TexturedVertexStreamBatchRange range{};
        range.vertex_offset = static_cast<u32>(master.vertices.size());
        range.vertex_count = batch.vertex_count;
        range.index_offset = static_cast<u32>(master.indices.size());
        range.index_count = batch.index_count;

        master.offsets.push_back(batch.offset);
        master.styles.push_back(TexturedVertexBatchStyle{batch.texture_id.bind_slot, batch.flags});
        const auto offset_result = gpu_data.offsets.push_back(master.offsets.back());
        const auto style_result = gpu_data.styles.push_back(master.styles.back());
        if (offset_result & agrb::vector_result_flag_bits::buffer_reallocated)
            gpu_data.descriptor_buffer_offsets_dirty = true;
        if (style_result & agrb::vector_result_flag_bits::buffer_reallocated)
            gpu_data.descriptor_buffer_styles_dirty = true;
        for (u32 i = 0; i < batch.vertex_count; ++i)
        {
            TexturedVertexStreamVertex vertex = batch.vertices[i];
            vertex.batch_id = static_cast<f32>(batch_id);
            master.vertices.push_back(vertex);
            gpu_data.vertices.push_back(vertex);
        }
        for (u32 i = 0; i < batch.index_count; ++i)
        {
            master.indices.push_back(batch.indices[i] + range.vertex_offset);
            gpu_data.indices.push_back(master.indices.back());
        }

        draw_data_id.render_id = batch_id;
        draw_data_id.hit_id = AUIK_INVALID_DRAW_DATA_ID;
        master.batches.push_back(range);
        const u32 vertices_version = next_cache_version(master.vertices_version);
        const u32 indices_version = next_cache_version(master.indices_version);
        const u32 offsets_version = next_cache_version(master.offsets_version);
        const u32 styles_version = next_cache_version(master.styles_version);
        reset_dirty_pages(master.vertices_pages, master.vertices.size(), vertices_version);
        reset_dirty_pages(master.indices_pages, master.indices.size(), indices_version);
        reset_dirty_pages(master.offsets_pages, master.offsets.size(), offsets_version);
        reset_dirty_pages(master.styles_pages, master.styles.size(), styles_version);
        gpu_data.vertices_version = vertices_version;
        gpu_data.indices_version = indices_version;
        gpu_data.offsets_version = offsets_version;
        gpu_data.styles_version = styles_version;
        gpu_data.vertices_pages = master.vertices_pages;
        gpu_data.indices_pages = master.indices_pages;
        gpu_data.offsets_pages = master.offsets_pages;
        gpu_data.styles_pages = master.styles_pages;
        ++stream->draw_sizes[frame_id];
        return draw_data_id;
    }

    static DrawDataID push_data_to_textured_vertex_stream(DrawStream *stream, const void *data, u32 frame_id)
    {
        return push_textured_vertex_stream_batch(stream, *static_cast<const TexturedVertexStreamBatchData *>(data),
                                                 frame_id);
    }

    static void push_data_batch_to_textured_vertex_stream(DrawStream *stream, const void *data, u32 count,
                                                          DrawDataID *out_draw_ids, u32 frame_id)
    {
        if (count == 0) return;
        const auto *batches = static_cast<const TexturedVertexStreamBatchData *>(data);
        for (u32 i = 0; i < count; ++i)
        {
            DrawDataID draw_id = push_textured_vertex_stream_batch(stream, batches[i], frame_id);
            if (out_draw_ids) out_draw_ids[i] = draw_id;
        }
    }

    static void update_textured_vertex_stream_batch(TexturedVertexStreamGPUData &gpu_data, u32 batch_id,
                                                    const TexturedVertexStreamBatchRange &range,
                                                    const TexturedVertexStreamBatchData &batch)
    {
        auto &master = *gpu_data.master;
        if (batch_id < master.offsets.size() && master.offsets[batch_id] != batch.offset)
        {
            master.offsets[batch_id] = batch.offset;
            gpu_data.offsets[batch_id] = batch.offset;
            const u32 version = next_cache_version(master.offsets_version);
            mark_dirty_page(master.offsets_pages, master.offsets.size(), batch_id, version);
            gpu_data.offsets_version = version;
            mark_dirty_page(gpu_data.offsets_pages, gpu_data.offsets.size(), batch_id, version);
        }
        if (!batch.vertices && !batch.indices && batch.vertex_count == 0u && batch.index_count == 0u) return;
        if ((!batch.vertices && batch.vertex_count > 0) || (!batch.indices && batch.index_count > 0)) return;
        if (batch.texture_id.handle == 0 || batch.texture_id.bind_slot == AUIK_INVALID_DRAW_DATA_ID) return;
        if (range.vertex_count != batch.vertex_count || range.index_count != batch.index_count) return;

        if (batch_id < master.styles.size())
        {
            const TexturedVertexBatchStyle style{batch.texture_id.bind_slot, batch.flags};
            const auto &current = master.styles[batch_id];
            if (current.texture_id != style.texture_id || current.flags != style.flags)
            {
                master.styles[batch_id] = style;
                gpu_data.styles[batch_id] = style;
                const u32 version = next_cache_version(master.styles_version);
                mark_dirty_page(master.styles_pages, master.styles.size(), batch_id, version);
                gpu_data.styles_version = version;
                mark_dirty_page(gpu_data.styles_pages, gpu_data.styles.size(), batch_id, version);
            }
        }

        for (u32 i = 0; i < batch.vertex_count; ++i)
        {
            TexturedVertexStreamVertex vertex = batch.vertices[i];
            vertex.batch_id = static_cast<f32>(batch_id);
            master.vertices[range.vertex_offset + i] = vertex;
            gpu_data.vertices[range.vertex_offset + i] = vertex;
        }
        for (u32 i = 0; i < batch.index_count; ++i)
        {
            const auto index = batch.indices[i] + range.vertex_offset;
            master.indices[range.index_offset + i] = index;
            gpu_data.indices[range.index_offset + i] = index;
        }
        const u32 vertices_version = next_cache_version(master.vertices_version);
        const u32 indices_version = next_cache_version(master.indices_version);
        mark_dirty_page_range(master.vertices_pages, master.vertices.size(), range.vertex_offset, range.vertex_count,
                              vertices_version);
        mark_dirty_page_range(master.indices_pages, master.indices.size(), range.index_offset, range.index_count,
                              indices_version);
        gpu_data.vertices_version = vertices_version;
        gpu_data.indices_version = indices_version;
        mark_dirty_page_range(gpu_data.vertices_pages, gpu_data.vertices.size(), range.vertex_offset,
                              range.vertex_count, vertices_version);
        mark_dirty_page_range(gpu_data.indices_pages, gpu_data.indices.size(), range.index_offset, range.index_count,
                              indices_version);
    }

    static void update_textured_vertex_stream_data(DrawStream *stream, DrawDataID draw_data_id, const void *data,
                                                   u32 frame_id)
    {
        auto &gpu_data = static_cast<TexturedVertexStreamGPUData *>(stream->stream_instances)[frame_id];
        auto &master = *gpu_data.master;
        if (draw_data_id.render_id >= master.batches.size()) return;
        update_textured_vertex_stream_batch(gpu_data, draw_data_id.render_id, master.batches[draw_data_id.render_id],
                                            *static_cast<const TexturedVertexStreamBatchData *>(data));
    }

    static void invalidate_textured_vertex_stream_data(DrawStream *stream, DrawDataID draw_data_id, u32 frame_id)
    {
        auto &gpu_data = static_cast<TexturedVertexStreamGPUData *>(stream->stream_instances)[frame_id];
        auto &master = *gpu_data.master;
        if (draw_data_id.render_id >= master.batches.size()) return;
        const auto &range = master.batches[draw_data_id.render_id];
        if (range.vertex_count == 0 || range.index_count == 0) return;

        const TexturedVertexStreamIndex degenerate_index = range.vertex_offset;
        for (u32 i = 0; i < range.index_count; ++i)
        {
            master.indices[range.index_offset + i] = degenerate_index;
            gpu_data.indices[range.index_offset + i] = degenerate_index;
        }
        const u32 version = next_cache_version(master.indices_version);
        mark_dirty_page_range(master.indices_pages, master.indices.size(), range.index_offset, range.index_count,
                              version);
        gpu_data.indices_version = version;
        mark_dirty_page_range(gpu_data.indices_pages, gpu_data.indices.size(), range.index_offset, range.index_count,
                              version);
    }

    static void update_textured_vertex_stream_data_batch(DrawStream *stream, const DrawDataID *draw_data_ids,
                                                         const void *data, u32 count, u32 frame_id)
    {
        if (count == 0) return;

        auto &gpu_data = static_cast<TexturedVertexStreamGPUData *>(stream->stream_instances)[frame_id];
        auto &master = *gpu_data.master;
        const auto *batches = static_cast<const TexturedVertexStreamBatchData *>(data);
        for (u32 i = 0; i < count; ++i)
        {
            const u32 render_id = draw_data_ids[i].render_id;
            if (render_id >= master.batches.size()) continue;
            update_textured_vertex_stream_batch(gpu_data, render_id, master.batches[render_id], batches[i]);
        }
    }

    static void clear_textured_vertex_stream(DrawStream *stream, u32 frame_id)
    {
        auto &gpu_data = static_cast<TexturedVertexStreamGPUData *>(stream->stream_instances)[frame_id];
        auto &master = *gpu_data.master;
        if (!master.vertices.empty()) next_cache_version(master.vertices_version);
        if (!master.indices.empty()) next_cache_version(master.indices_version);
        if (!master.offsets.empty()) next_cache_version(master.offsets_version);
        if (!master.styles.empty()) next_cache_version(master.styles_version);
        master.vertices.clear();
        master.indices.clear();
        master.offsets.clear();
        master.styles.clear();
        master.batches.clear();
        reset_dirty_pages(master.vertices_pages, 0u, master.vertices_version);
        reset_dirty_pages(master.indices_pages, 0u, master.indices_version);
        reset_dirty_pages(master.offsets_pages, 0u, master.offsets_version);
        reset_dirty_pages(master.styles_pages, 0u, master.styles_version);
        gpu_data.vertices.clear();
        gpu_data.indices.clear();
        gpu_data.offsets.clear();
        gpu_data.styles.clear();
        gpu_data.vertices_version = master.vertices_version;
        gpu_data.indices_version = master.indices_version;
        gpu_data.offsets_version = master.offsets_version;
        gpu_data.styles_version = master.styles_version;
        gpu_data.vertices_pages = master.vertices_pages;
        gpu_data.indices_pages = master.indices_pages;
        gpu_data.offsets_pages = master.offsets_pages;
        gpu_data.styles_pages = master.styles_pages;
    }

    static void copy_textured_vertex_stream_frame_data(DrawStream *stream, u32 dst_frame_id, u32 src_frame_id)
    {
        if (dst_frame_id == src_frame_id) return;

        auto *frames = static_cast<TexturedVertexStreamGPUData *>(stream->stream_instances);
        auto &dst = frames[dst_frame_id];
        auto &master = *dst.master;
        const size_t vertex_count = master.vertices.size();
        const size_t index_count = master.indices.size();
        const size_t batch_count = master.offsets.size();
        const bool vertices_size_changed = dst.vertices.size() != vertex_count;
        const bool indices_size_changed = dst.indices.size() != index_count;
        const bool offsets_size_changed = dst.offsets.size() != batch_count;
        const bool styles_size_changed = dst.styles.size() != master.styles.size();

        if (dst.vertices.size() != vertex_count)
        {
            dst.vertices.clear();
            if (!(dst.vertices.resize(vertex_count) & agrb::vector_result_flag_bits::success))
            {
                stream->draw_sizes[dst_frame_id] = 0u;
                return;
            }
        }
        if (dst.indices.size() != index_count)
        {
            dst.indices.clear();
            if (!(dst.indices.resize(index_count) & agrb::vector_result_flag_bits::success))
            {
                stream->draw_sizes[dst_frame_id] = 0u;
                return;
            }
        }
        if (dst.offsets.size() != batch_count)
        {
            dst.offsets.clear();
            const auto result = dst.offsets.resize(batch_count);
            if (!(result & agrb::vector_result_flag_bits::success))
            {
                stream->draw_sizes[dst_frame_id] = 0u;
                return;
            }
            if (result & agrb::vector_result_flag_bits::buffer_reallocated) dst.descriptor_buffer_offsets_dirty = true;
        }
        if (dst.styles.size() != master.styles.size())
        {
            dst.styles.clear();
            const auto result = dst.styles.resize(master.styles.size());
            if (!(result & agrb::vector_result_flag_bits::success))
            {
                stream->draw_sizes[dst_frame_id] = 0u;
                return;
            }
            if (result & agrb::vector_result_flag_bits::buffer_reallocated) dst.descriptor_buffer_styles_dirty = true;
        }

        if (dst.vertices_version != master.vertices_version || vertices_size_changed)
            upload_dirty_pages(dst.vertices, master.vertices, dst.vertices_pages, master.vertices_pages,
                               vertices_size_changed);
        if (dst.indices_version != master.indices_version || indices_size_changed)
            upload_dirty_pages(dst.indices, master.indices, dst.indices_pages, master.indices_pages,
                               indices_size_changed);
        if (dst.offsets_version != master.offsets_version || offsets_size_changed)
            upload_dirty_pages(dst.offsets, master.offsets, dst.offsets_pages, master.offsets_pages,
                               offsets_size_changed);
        if (dst.styles_version != master.styles_version || styles_size_changed)
            upload_dirty_pages(dst.styles, master.styles, dst.styles_pages, master.styles_pages, styles_size_changed);
        dst.vertices_version = master.vertices_version;
        dst.indices_version = master.indices_version;
        dst.offsets_version = master.offsets_version;
        dst.styles_version = master.styles_version;
        stream->draw_sizes[dst_frame_id] = static_cast<u32>(master.batches.size());
    }

    static void destroy_textured_vertex_stream_gpu_data(DrawStream *stream)
    {
        const u32 count = get_context().frames_in_flight;
        TexturedVertexStreamMasterData *master = nullptr;
        for (u32 i = 0; i < count; ++i)
        {
            auto &gpu_data = static_cast<TexturedVertexStreamGPUData *>(stream->stream_instances)[i];
            if (!master) master = gpu_data.master;
            gpu_data.vertices.destroy();
            gpu_data.indices.destroy();
            gpu_data.offsets.destroy();
            gpu_data.styles.destroy();
        }
        acul::release(master);
        auto *frames = static_cast<TexturedVertexStreamGPUData *>(stream->stream_instances);
        acul::release(frames, count);
    }

    static void *create_textured_vertex_stream_gpu_data(u32 instance_count, GPUContext *gpu_context)
    {
        auto *data = acul::alloc_n<TexturedVertexStreamGPUData>(instance_count);
        auto *master = acul::alloc<TexturedVertexStreamMasterData>();
        auto &device = get_agrb_device(gpu_context);
        agrb::managed_buffer vertex_buf{.required_flags = vk::MemoryPropertyFlagBits::eHostVisible |
                                                          vk::MemoryPropertyFlagBits::eHostCoherent,
                                        .buffer_usage = vk::BufferUsageFlagBits::eVertexBuffer,
                                        .vma_usage = VMA_MEMORY_USAGE_CPU_TO_GPU};
        agrb::managed_buffer index_buf{.required_flags = vk::MemoryPropertyFlagBits::eHostVisible |
                                                         vk::MemoryPropertyFlagBits::eHostCoherent,
                                       .buffer_usage = vk::BufferUsageFlagBits::eIndexBuffer,
                                       .vma_usage = VMA_MEMORY_USAGE_CPU_TO_GPU};
        agrb::managed_buffer batch_buf{.required_flags = vk::MemoryPropertyFlagBits::eHostVisible |
                                                         vk::MemoryPropertyFlagBits::eHostCoherent,
                                       .buffer_usage = vk::BufferUsageFlagBits::eStorageBuffer,
                                       .vma_usage = VMA_MEMORY_USAGE_CPU_TO_GPU};
        for (u32 i = 0; i < instance_count; ++i)
        {
            data[i].master = master;
            data[i].vertices.init(device, vertex_buf);
            data[i].indices.init(device, index_buf);
            data[i].offsets.init(device, batch_buf);
            data[i].styles.init(device, batch_buf);
        }
        return data;
    }

    static bool update_textured_vertex_stream_descriptor_set(DrawStream *stream, TexturedVertexStreamGPUData &gpu_data,
                                                             GPUContext *gpu_context, u32 frame_id)
    {
        auto *pipeline = stream->pipeline;
        auto *ctx = get_agrb_context(gpu_context);
        assert(pipeline && pipeline->descriptor_set_layout);
        assert(ctx->clip_rects);

        const auto &clip_rects_data = ctx->clip_rects[frame_id].data();
        const vk::Buffer clip_rects_buffer = clip_rects_data.vk_buffer;
        const vk::Buffer offsets_buffer = gpu_data.offsets.data().vk_buffer;
        const vk::Buffer styles_buffer = gpu_data.styles.data().vk_buffer;
        const bool clip_rects_reallocated = ctx->clip_rects_reallocated && ctx->clip_rects_reallocated[frame_id];
        if (!clip_rects_buffer || !offsets_buffer || !styles_buffer) return false;
        if (gpu_data.descriptor_set && gpu_data.descriptor_buffer_clip_rects == clip_rects_buffer &&
            gpu_data.descriptor_buffer_offsets == offsets_buffer &&
            gpu_data.descriptor_buffer_styles == styles_buffer && !clip_rects_reallocated &&
            !gpu_data.descriptor_buffer_offsets_dirty && !gpu_data.descriptor_buffer_styles_dirty)
            return true;

        vk::DescriptorBufferInfo clip_rects_info{clip_rects_buffer, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo offsets_info{offsets_buffer, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo styles_info{styles_buffer, 0, VK_WHOLE_SIZE};
        agrb::descriptor_writer writer(*pipeline->descriptor_set_layout, *ctx->descriptor_pool);
        writer.write_buffer(0, &clip_rects_info);
        writer.write_buffer(1, &offsets_info);
        writer.write_buffer(2, &styles_info);
        if (!gpu_data.descriptor_set)
        {
            if (!writer.build(gpu_data.descriptor_set)) return false;
        }
        else writer.overwrite(gpu_data.descriptor_set);

        gpu_data.descriptor_buffer_clip_rects = clip_rects_buffer;
        gpu_data.descriptor_buffer_offsets = offsets_buffer;
        gpu_data.descriptor_buffer_styles = styles_buffer;
        gpu_data.descriptor_buffer_offsets_dirty = false;
        gpu_data.descriptor_buffer_styles_dirty = false;
        return true;
    }

    static void render_textured_vertex_stream(DrawStream *stream, void *render_ctx, GPUContext *gpu_context,
                                              u32 frame_id)
    {
        auto &gpu_data = static_cast<TexturedVertexStreamGPUData *>(stream->stream_instances)[frame_id];
        if (gpu_data.indices.empty() || gpu_data.vertices.empty() || stream->draw_sizes[frame_id] == 0u) return;
        auto *ctx = get_agrb_context(gpu_context);
        if (!ctx->bindless_texture_set) return;
        if (!update_textured_vertex_stream_descriptor_set(stream, gpu_data, gpu_context, frame_id)) return;

        auto *pipeline = stream->pipeline;
        auto &device = get_agrb_device(gpu_context);
        auto &cmd = *static_cast<vk::CommandBuffer *>(render_ctx);
        auto &loader = device.loader;
        const vk::Buffer vertex_buffers[] = {gpu_data.vertices.data().vk_buffer};
        const vk::DeviceSize offsets[] = {0};
        const vk::DescriptorSet descriptor_sets[] = {gpu_data.descriptor_set, ctx->bindless_texture_set};
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->handle, loader);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline->layout, 0, 2, descriptor_sets, 0, nullptr,
                               loader);
        cmd.bindVertexBuffers(0, 1, vertex_buffers, offsets, loader);
        cmd.bindIndexBuffer(gpu_data.indices.data().vk_buffer, 0, vk::IndexType::eUint32, loader);

        const amal::vec2 push_data = get_display_size();
        cmd.pushConstants(pipeline->layout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(push_data), &push_data, loader);
        cmd.drawIndexed(static_cast<u32>(gpu_data.indices.size()), 1, 0, 0, 0, loader);
    }

    void init_textured_vertex_stream_pipeline_calls(StreamGPUDispatch &dispatch)
    {
        dispatch.push_data_to_stream = &push_data_to_textured_vertex_stream;
        dispatch.push_data_batch_to_stream = &push_data_batch_to_textured_vertex_stream;
        dispatch.update_stream_data = &update_textured_vertex_stream_data;
        dispatch.update_stream_data_batch = &update_textured_vertex_stream_data_batch;
        dispatch.invalidate_stream_data = &invalidate_textured_vertex_stream_data;
        dispatch.clear_stream = &clear_textured_vertex_stream;
        dispatch.copy_stream_frame_data = &copy_textured_vertex_stream_frame_data;
        dispatch.render_stream = &render_textured_vertex_stream;
        dispatch.create_stream_gpu_data = &create_textured_vertex_stream_gpu_data;
        dispatch.destroy_stream_gpu_data = &destroy_textured_vertex_stream_gpu_data;
    }
} // namespace auik::detail

namespace auik
{
    bool construct_textured_vertex_pipeline(DrawPipeline &pipeline, agrb::device &device)
    {
        pipeline.descriptor_set_layout =
            agrb::descriptor_set_layout::builder()
                .add_binding(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eFragment)
                .add_binding(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                .add_binding(2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                .build(device);
        if (!pipeline.descriptor_set_layout) return false;

        auto *ctx = detail::get_agrb_context(detail::get_context().gpu_ctx);
        assert(ctx && ctx->bindless_texture_layout);
        const vk::DescriptorSetLayout set_layouts[] = {pipeline.descriptor_set_layout->layout(),
                                                       ctx->bindless_texture_layout->layout()};
        const vk::PushConstantRange push_constant{vk::ShaderStageFlagBits::eVertex, 0, sizeof(amal::vec2)};
        vk::PipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.setLayoutCount = 2;
        pipeline_layout_info.pSetLayouts = set_layouts;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant;
        pipeline.layout = device.vk_device.createPipelineLayout(pipeline_layout_info, nullptr, device.loader);
        return pipeline.layout != nullptr;
    }

    bool configure_textured_vertex_pipeline(agrb::graphics_pipeline_batch::artifact &artifact,
                                            vk::RenderPass render_pass, DrawPipeline &pipeline, agrb::device &device)
    {
        auto *tmp = static_cast<agrb::graphics_pipeline_batch::artifact::custom_data_t<u32> *>(artifact.tmp);
        if (!tmp) return false;

        artifact.config.load_defaults().enable_alpha_blending();
        auto &blend = artifact.config.color_blend_attachment;
        blend.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
        blend.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        blend.colorBlendOp = vk::BlendOp::eAdd;
        blend.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        blend.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        blend.alphaBlendOp = vk::BlendOp::eAdd;
        artifact.config.depth_stencil_info.setDepthTestEnable(true).setDepthWriteEnable(true).setDepthCompareOp(
            vk::CompareOp::eGreaterOrEqual);
        artifact.config.render_pass = render_pass;
        artifact.config.pipeline_layout = pipeline.layout;
        artifact.config.subpass = tmp->value;
        artifact.config.binding_descriptions = {
            vk::VertexInputBindingDescription{0, sizeof(TexturedVertexStreamVertex), vk::VertexInputRate::eVertex}};
        artifact.config.attribute_descriptions = {
            vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32B32A32Sfloat,
                                                static_cast<u32>(offsetof(TexturedVertexStreamVertex, position))},
            vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32Sfloat,
                                                static_cast<u32>(offsetof(TexturedVertexStreamVertex, uv))},
            vk::VertexInputAttributeDescription{2, 0, vk::Format::eR32Uint,
                                                static_cast<u32>(offsetof(TexturedVertexStreamVertex, clip_id))}};
        artifact.config.vertex_input_info
            .setVertexBindingDescriptionCount(static_cast<u32>(artifact.config.binding_descriptions.size()))
            .setPVertexBindingDescriptions(artifact.config.binding_descriptions.data())
            .setVertexAttributeDescriptionCount(static_cast<u32>(artifact.config.attribute_descriptions.size()))
            .setPVertexAttributeDescriptions(artifact.config.attribute_descriptions.data());

        auto *ctx = detail::get_agrb_context(detail::get_context().gpu_ctx);
        if (!ctx) return false;

        const auto &path = detail::get_shader_library_path();
        vk::ShaderModule shaders[2];
        auto vs = ctx->shader_cache.get_shader(AS_AUIK_TEXTURED_VERTEX_STREAM_VS, shaders[0], device, path);
        if (!vs.success()) return false;
        auto fs = ctx->shader_cache.get_shader(AS_AUIK_TEXTURED_VERTEX_STREAM_FS, shaders[1], device, path);
        if (!fs.success()) return false;
        agrb::prepare_base_graphics_pipeline(artifact, shaders, device);
        return true;
    }
} // namespace auik
