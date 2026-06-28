#include <cstddef>
#include <acul/scalars.hpp>
#include <acul/string/string.hpp>
#include <acul/vector.hpp>
#include <auik-gbe-agrb/shaders.h>
#include <auik/gbe/agrb/vertex_pipeline.hpp>
#include <auik/pipelines.hpp>
#include "../context.hpp"

namespace auik::detail
{
    struct VertexStreamBatchRange
    {
        u32 vertex_offset = 0;
        u32 vertex_count = 0;
        u32 index_offset = 0;
        u32 index_count = 0;
    };

    struct VertexStreamGPUData
    {
        agrb::vector<VertexStreamVertex> vertices;
        agrb::vector<VertexStreamIndex> indices;
        acul::vector<VertexStreamBatchRange> batches;
        vk::DescriptorSet descriptor_set;
        vk::Buffer descriptor_buffer_clip_rects = nullptr;
    };

    static DrawDataID push_vertex_stream_batch(DrawStream *stream, const VertexStreamBatchData &batch, u32 frame_id)
    {
        DrawDataID draw_data_id{};
        if ((!batch.vertices && batch.vertex_count > 0) || (!batch.indices && batch.index_count > 0)) return draw_data_id;

        auto &gpu_data = static_cast<VertexStreamGPUData *>(stream->stream_instances)[frame_id];
        VertexStreamBatchRange range{};
        range.vertex_offset = static_cast<u32>(gpu_data.vertices.size());
        range.vertex_count = batch.vertex_count;
        range.index_offset = static_cast<u32>(gpu_data.indices.size());
        range.index_count = batch.index_count;

        for (u32 i = 0; i < batch.vertex_count; ++i) gpu_data.vertices.push_back(batch.vertices[i]);
        for (u32 i = 0; i < batch.index_count; ++i) gpu_data.indices.push_back(batch.indices[i] + range.vertex_offset);

        draw_data_id.render_id = static_cast<u32>(gpu_data.batches.size());
        draw_data_id.hit_id = AUIK_INVALID_DRAW_DATA_ID;
        gpu_data.batches.push_back(range);
        ++stream->draw_sizes[frame_id];
        return draw_data_id;
    }

    static DrawDataID push_data_to_vertex_stream(DrawStream *stream, const void *data, u32 frame_id)
    {
        return push_vertex_stream_batch(stream, *static_cast<const VertexStreamBatchData *>(data), frame_id);
    }

    static void push_data_batch_to_vertex_stream(DrawStream *stream, const void *data, u32 count, DrawDataID *out_draw_ids,
                                                 u32 frame_id)
    {
        if (count == 0) return;
        const auto *batches = static_cast<const VertexStreamBatchData *>(data);
        for (u32 i = 0; i < count; ++i)
        {
            DrawDataID draw_id = push_vertex_stream_batch(stream, batches[i], frame_id);
            if (out_draw_ids) out_draw_ids[i] = draw_id;
        }
    }

    static void update_vertex_stream_batch(VertexStreamGPUData &gpu_data, const VertexStreamBatchRange &range,
                                           const VertexStreamBatchData &batch)
    {
        if ((!batch.vertices && batch.vertex_count > 0) || (!batch.indices && batch.index_count > 0)) return;
        if (range.vertex_count != batch.vertex_count || range.index_count != batch.index_count) return;

        for (u32 i = 0; i < batch.vertex_count; ++i) gpu_data.vertices[range.vertex_offset + i] = batch.vertices[i];
        for (u32 i = 0; i < batch.index_count; ++i)
            gpu_data.indices[range.index_offset + i] = batch.indices[i] + range.vertex_offset;
    }

    static void update_vertex_stream_data(DrawStream *stream, DrawDataID draw_data_id, const void *data, u32 frame_id)
    {
        auto &gpu_data = static_cast<VertexStreamGPUData *>(stream->stream_instances)[frame_id];
        if (draw_data_id.render_id >= gpu_data.batches.size()) return;
        update_vertex_stream_batch(gpu_data, gpu_data.batches[draw_data_id.render_id],
                                   *static_cast<const VertexStreamBatchData *>(data));
    }

    static void invalidate_vertex_stream_data(DrawStream *stream, DrawDataID draw_data_id, u32 frame_id)
    {
        auto &gpu_data = static_cast<VertexStreamGPUData *>(stream->stream_instances)[frame_id];
        if (draw_data_id.render_id >= gpu_data.batches.size()) return;
        const auto &range = gpu_data.batches[draw_data_id.render_id];
        if (range.vertex_count == 0 || range.index_count == 0) return;

        const VertexStreamIndex degenerate_index = range.vertex_offset;
        for (u32 i = 0; i < range.index_count; ++i) gpu_data.indices[range.index_offset + i] = degenerate_index;
    }

    static void update_vertex_stream_data_batch(DrawStream *stream, const DrawDataID *draw_data_ids, const void *data,
                                                u32 count, u32 frame_id)
    {
        if (count == 0) return;

        auto &gpu_data = static_cast<VertexStreamGPUData *>(stream->stream_instances)[frame_id];
        const auto *batches = static_cast<const VertexStreamBatchData *>(data);
        for (u32 i = 0; i < count; ++i)
        {
            const u32 render_id = draw_data_ids[i].render_id;
            if (render_id >= gpu_data.batches.size()) continue;
            update_vertex_stream_batch(gpu_data, gpu_data.batches[render_id], batches[i]);
        }
    }

    static void clear_vertex_stream(DrawStream *stream, u32 frame_id)
    {
        auto &gpu_data = static_cast<VertexStreamGPUData *>(stream->stream_instances)[frame_id];
        gpu_data.vertices.clear();
        gpu_data.indices.clear();
        gpu_data.batches.clear();
    }

    static void copy_vertex_stream_frame_data(DrawStream *stream, u32 dst_frame_id, u32 src_frame_id)
    {
        if (dst_frame_id == src_frame_id) return;

        auto *frames = static_cast<VertexStreamGPUData *>(stream->stream_instances);
        auto &dst = frames[dst_frame_id];
        auto &src = frames[src_frame_id];
        dst.vertices.clear();
        dst.indices.clear();
        for (u32 i = 0; i < src.vertices.size(); ++i) dst.vertices.push_back(src.vertices[i]);
        for (u32 i = 0; i < src.indices.size(); ++i) dst.indices.push_back(src.indices[i]);
        dst.batches.clear();
        dst.batches.insert(dst.batches.end(), src.batches.begin(), src.batches.end());
        stream->draw_sizes[dst_frame_id] = stream->draw_sizes[src_frame_id];
    }

    static void destroy_vertex_stream_gpu_data(DrawStream *stream)
    {
        const u32 count = get_context().frames_in_flight;
        for (u32 i = 0; i < count; ++i)
        {
            auto &gpu_data = static_cast<VertexStreamGPUData *>(stream->stream_instances)[i];
            gpu_data.vertices.destroy();
            gpu_data.indices.destroy();
            gpu_data.batches.clear();
        }
        acul::release(static_cast<VertexStreamGPUData *>(stream->stream_instances), count);
    }

    static void *create_vertex_stream_gpu_data(u32 instance_count, GPUContext *gpu_context)
    {
        auto *data = acul::alloc_n<VertexStreamGPUData>(instance_count);
        auto &device = get_agrb_device(gpu_context);
        agrb::managed_buffer vertex_buf{
            .required_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            .buffer_usage = vk::BufferUsageFlagBits::eVertexBuffer,
            .vma_usage = VMA_MEMORY_USAGE_CPU_TO_GPU};
        agrb::managed_buffer index_buf{
            .required_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            .buffer_usage = vk::BufferUsageFlagBits::eIndexBuffer,
            .vma_usage = VMA_MEMORY_USAGE_CPU_TO_GPU};
        for (u32 i = 0; i < instance_count; ++i)
        {
            data[i].vertices.init(device, vertex_buf);
            data[i].indices.init(device, index_buf);
        }
        return data;
    }

    static bool update_vertex_stream_descriptor_set(DrawStream *stream, VertexStreamGPUData &gpu_data,
                                                    GPUContext *gpu_context, u32 frame_id)
    {
        auto *pipeline = stream->pipeline;
        auto *ctx = get_agrb_context(gpu_context);
        assert(pipeline && pipeline->descriptor_set_layout);
        assert(ctx->clip_rects);

        const auto &clip_rects_data = ctx->clip_rects[frame_id].data();
        const vk::Buffer clip_rects_buffer = clip_rects_data.vk_buffer;
        const bool clip_rects_reallocated = ctx->clip_rects_reallocated && ctx->clip_rects_reallocated[frame_id];
        if (!clip_rects_buffer) return false;
        if (gpu_data.descriptor_set && gpu_data.descriptor_buffer_clip_rects == clip_rects_buffer &&
            !clip_rects_reallocated)
            return true;

        vk::DescriptorBufferInfo clip_rects_info{clip_rects_buffer, 0, VK_WHOLE_SIZE};
        agrb::descriptor_writer writer(*pipeline->descriptor_set_layout, *ctx->descriptor_pool);
        writer.write_buffer(0, &clip_rects_info);
        if (!gpu_data.descriptor_set)
        {
            if (!writer.build(gpu_data.descriptor_set)) return false;
        }
        else writer.overwrite(gpu_data.descriptor_set);

        gpu_data.descriptor_buffer_clip_rects = clip_rects_buffer;
        return true;
    }

    static void render_vertex_stream(DrawStream *stream, void *render_ctx, GPUContext *gpu_context, u32 frame_id)
    {
        auto &gpu_data = static_cast<VertexStreamGPUData *>(stream->stream_instances)[frame_id];
        if (gpu_data.indices.empty() || gpu_data.vertices.empty()) return;
        if (!update_vertex_stream_descriptor_set(stream, gpu_data, gpu_context, frame_id)) return;

        auto *pipeline = stream->pipeline;
        auto &device = get_agrb_device(gpu_context);
        auto &cmd = *static_cast<vk::CommandBuffer *>(render_ctx);
        auto &loader = device.loader;
        const vk::Buffer vertex_buffers[] = {gpu_data.vertices.data().vk_buffer};
        const vk::DeviceSize offsets[] = {0};
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->handle, loader);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline->layout, 0, 1, &gpu_data.descriptor_set, 0,
                               nullptr, loader);
        cmd.bindVertexBuffers(0, 1, vertex_buffers, offsets, loader);
        cmd.bindIndexBuffer(gpu_data.indices.data().vk_buffer, 0, vk::IndexType::eUint32, loader);
        cmd.pushConstants(pipeline->layout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(amal::vec2),
                          &get_display_size(), loader);
        cmd.drawIndexed(static_cast<u32>(gpu_data.indices.size()), 1, 0, 0, 0, loader);
    }

    void init_vertex_stream_pipeline_calls(StreamGPUDispatch &dispatch)
    {
        dispatch.push_data_to_stream = &push_data_to_vertex_stream;
        dispatch.push_data_batch_to_stream = &push_data_batch_to_vertex_stream;
        dispatch.update_stream_data = &update_vertex_stream_data;
        dispatch.update_stream_data_batch = &update_vertex_stream_data_batch;
        dispatch.invalidate_stream_data = &invalidate_vertex_stream_data;
        dispatch.clear_stream = &clear_vertex_stream;
        dispatch.copy_stream_frame_data = &copy_vertex_stream_frame_data;
        dispatch.render_stream = &render_vertex_stream;
        dispatch.create_stream_gpu_data = &create_vertex_stream_gpu_data;
        dispatch.destroy_stream_gpu_data = &destroy_vertex_stream_gpu_data;
    }
} // namespace auik::detail

namespace auik
{
    bool construct_vertex_pipeline(DrawPipeline &pipeline, agrb::device &device)
    {
        pipeline.descriptor_set_layout =
            agrb::descriptor_set_layout::builder()
                .add_binding(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eFragment)
                .build(device);
        if (!pipeline.descriptor_set_layout) return false;

        const vk::DescriptorSetLayout set_layouts[] = {pipeline.descriptor_set_layout->layout()};
        const vk::PushConstantRange push_constant{vk::ShaderStageFlagBits::eVertex, 0, sizeof(amal::vec2)};
        vk::PipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = set_layouts;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant;
        pipeline.layout = device.vk_device.createPipelineLayout(pipeline_layout_info, nullptr, device.loader);
        return pipeline.layout != nullptr;
    }

    bool configure_vertex_pipeline(agrb::graphics_pipeline_batch::artifact &artifact, vk::RenderPass render_pass,
                                   DrawPipeline &pipeline, agrb::device &device)
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
            vk::VertexInputBindingDescription{0, sizeof(VertexStreamVertex), vk::VertexInputRate::eVertex}};
        artifact.config.attribute_descriptions = {
            vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32B32A32Sfloat,
                                                static_cast<u32>(offsetof(VertexStreamVertex, position))},
            vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32Uint,
                                                static_cast<u32>(offsetof(VertexStreamVertex, color))}};
        artifact.config.vertex_input_info.setVertexBindingDescriptionCount(static_cast<u32>(
                                              artifact.config.binding_descriptions.size()))
            .setPVertexBindingDescriptions(artifact.config.binding_descriptions.data())
            .setVertexAttributeDescriptionCount(static_cast<u32>(artifact.config.attribute_descriptions.size()))
            .setPVertexAttributeDescriptions(artifact.config.attribute_descriptions.data());

        auto *ctx = detail::get_agrb_context(detail::get_context().gpu_ctx);
        if (!ctx) return false;

        const auto &path = detail::get_shader_library_path();
        vk::ShaderModule shaders[2];
        auto vs = ctx->shader_cache.get_shader(AS_AUIK_VERTEX_STREAM_VS, shaders[0], device, path);
        if (!vs.success()) return false;
        auto fs = ctx->shader_cache.get_shader(AS_AUIK_VERTEX_STREAM_FS, shaders[1], device, path);
        if (!fs.success()) return false;
        agrb::prepare_base_graphics_pipeline(artifact, shaders, device);
        return true;
    }
} // namespace auik
