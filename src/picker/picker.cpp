#include "picker.hpp"
#include <agrb/defaults.hpp>
#include <agrb/utils/buffer.hpp>
#include <agrb/utils/image.hpp>
#include <auik-gbe-agrb/shaders.h>
#include <auik/detail/context.hpp>
#include <auik/gbe/agrb/agrb.hpp>
#include "../context.hpp"

namespace auik::detail
{
    static constexpr vk::DeviceSize AUIK_PICK_RESULT_SIZE = sizeof(u32) * 4;

    void GPUPicker::create_render_pass(agrb::device &device)
    {
        vk::AttachmentDescription2 attachments[2];
        attachments[0]
            .setFormat(vk::Format::eR32G32B32A32Uint)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::eTransferSrcOptimal);
        attachments[1]
            .setFormat(_depth_format)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

        vk::AttachmentReference2 color_ref;
        color_ref.setAttachment(0)
            .setLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setAspectMask(vk::ImageAspectFlagBits::eColor);
        vk::AttachmentReference2 depth_ref;
        depth_ref.setAttachment(1)
            .setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
            .setAspectMask(vk::ImageAspectFlagBits::eDepth);

        vk::SubpassDescription2 subpass;
        subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
            .setColorAttachmentCount(1)
            .setPColorAttachments(&color_ref)
            .setPDepthStencilAttachment(&depth_ref);

        vk::SubpassDependency2 deps[2];
        deps[0]
            .setSrcSubpass(vk::SubpassExternal)
            .setDstSubpass(0)
            .setSrcStageMask(vk::PipelineStageFlagBits::eTopOfPipe)
            .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput |
                             vk::PipelineStageFlagBits::eEarlyFragmentTests |
                             vk::PipelineStageFlagBits::eLateFragmentTests)
            .setSrcAccessMask(vk::AccessFlagBits::eNone)
            .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite |
                              vk::AccessFlagBits::eDepthStencilAttachmentWrite)
            .setDependencyFlags(vk::DependencyFlagBits::eByRegion);
        deps[1]
            .setSrcSubpass(0)
            .setDstSubpass(vk::SubpassExternal)
            .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput |
                             vk::PipelineStageFlagBits::eLateFragmentTests)
            .setDstStageMask(vk::PipelineStageFlagBits::eTransfer)
            .setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite |
                              vk::AccessFlagBits::eDepthStencilAttachmentWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead)
            .setDependencyFlags(vk::DependencyFlagBits::eByRegion);

        vk::RenderPassCreateInfo2 ci;
        ci.setAttachmentCount(2)
            .setPAttachments(attachments)
            .setSubpassCount(1)
            .setPSubpasses(&subpass)
            .setDependencyCount(2)
            .setPDependencies(deps);
        rp_group.value(device.vk_device.createRenderPass2(ci, nullptr, device.loader));
    }

    static bool create_color_image(agrb::fb_image_slot &slot, vk::Extent2D extent, agrb::device &device)
    {
        auto &image = slot.attachments[0];
        vk::ImageCreateInfo image_info;
        image_info.setImageType(vk::ImageType::e2D)
            .setExtent({1, 1, 1})
            .setMipLevels(1)
            .setArrayLayers(1)
            .setFormat(vk::Format::eR32G32B32A32Uint)
            .setTiling(vk::ImageTiling::eOptimal)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setSharingMode(vk::SharingMode::eExclusive);
        auto create_info = agrb::make_alloc_info(VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, {},
                                                 vk::MemoryPropertyFlagBits::eDeviceLocal, 1.0f);
        if (!agrb::create_image(image_info, image.image, image.memory, device.allocator, create_info)) return false;

        vk::ImageViewCreateInfo view_info;
        view_info.setImage(image.image)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(vk::Format::eR32G32B32A32Uint)
            .setSubresourceRange(agrb::defaults::subresource_range_color);
        return device.vk_device.createImageView(&view_info, nullptr, &image.get_view(), device.loader) ==
               vk::Result::eSuccess;
    }

    static bool create_depth_image(agrb::fb_image_slot &slot, vk::Extent2D extent, vk::Format image_format,
                                   agrb::device &device)
    {
        auto &image = slot.attachments[1];
        vk::ImageCreateInfo image_info;
        image_info.setImageType(vk::ImageType::e2D)
            .setExtent({1, 1, 1})
            .setMipLevels(1)
            .setArrayLayers(1)
            .setFormat(image_format)
            .setTiling(vk::ImageTiling::eOptimal)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setSharingMode(vk::SharingMode::eExclusive);
        auto create_info = agrb::make_alloc_info(VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, {},
                                                 vk::MemoryPropertyFlagBits::eDeviceLocal, 1.0f);
        if (!agrb::create_image(image_info, image.image, image.memory, device.allocator, create_info)) return false;

        vk::ImageViewCreateInfo view_info;
        view_info.setImage(image.image)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(image_format)
            .setSubresourceRange(agrb::defaults::subresource_range_depth);
        return device.vk_device.createImageView(&view_info, nullptr, &image.get_view(), device.loader) ==
               vk::Result::eSuccess;
    }

    bool GPUPicker::create_attachments(agrb::device &device)
    {
        u32 frames_in_flight = get_context().frames_in_flight;
        attachments = acul::alloc<agrb::fb_attachments>(vk::Extent2D{1, 1});
        attachments->attachment_count = 2;
        attachments->image_count = frames_in_flight;
        attachments->images = acul::alloc_n<agrb::fb_image_slot>(frames_in_flight);

        for (u32 i = 0; i < frames_in_flight; ++i)
        {
            auto &slot = attachments->images[i];
            slot.attachments.resize(2);

            if (!create_color_image(slot, attachments->extent, device)) return false;
            if (!create_depth_image(slot, attachments->extent, _depth_format, device)) return false;
        }

        if (!clear_values)
        {
            clear_values = acul::alloc_n<vk::ClearValue>(2);
            clear_values[0] = vk::ClearValue(vk::ClearColorValue(std::array<u32, 4>{0u, 0u, 0u, 0u}));
            clear_values[1] = vk::ClearValue(vk::ClearDepthStencilValue(0.0f, 0));
        }

        return agrb::create_fb_handles(this, device);
    }

    bool GPUPicker::create_descriptor_resources(agrb::device &device)
    {
        agrb::managed_buffer buf{
            .required_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            .buffer_usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                            vk::BufferUsageFlagBits::eTransferDst,
            .vma_usage = VMA_MEMORY_USAGE_CPU_TO_GPU};
        buf.instance_count = 1;
        const u32 frames_in_flight = get_context().frames_in_flight;
        _rects = acul::alloc_n<PickerFrameData>(frames_in_flight);
        for (u32 i = 0; i < frames_in_flight; ++i)
        {
            _rects[i].transforms.init(device, buf);
            _rects[i].styles.init(device, buf);
        }

        _descriptor_set_layout =
            agrb::descriptor_set_layout::builder()
                .add_binding(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                .add_binding(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                .add_binding(2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eFragment)
                .build(device);
        if (!_descriptor_set_layout) return false;
        auto &global_ctx = get_context();
        _descriptor_sets.resize(frames_in_flight);
        _descriptor_buffer_transforms.resize(frames_in_flight);
        _descriptor_buffer_styles.resize(frames_in_flight);
        _descriptor_buffer_clip_rects.resize(frames_in_flight);
        _descriptor_buffer_transforms_dirty.resize(frames_in_flight);
        _descriptor_buffer_styles_dirty.resize(frames_in_flight);

        auto *agrb_ctx = get_agrb_context(global_ctx.gpu_ctx);
        if (!agrb_ctx->clip_rects) return false;
        for (u32 i = 0; i < frames_in_flight; ++i)
        {
            const auto &transforms_data = _rects[i].transforms.data();
            const auto &styles_data = _rects[i].styles.data();
            const auto &clip_rects_data = agrb_ctx->clip_rects[i].data();
            const vk::Buffer transform_buffer = transforms_data.vk_buffer;
            const vk::Buffer style_buffer = styles_data.vk_buffer;
            const vk::Buffer clip_rects_buffer = clip_rects_data.vk_buffer;
            if (!transform_buffer || !style_buffer || !clip_rects_buffer) return false;
            vk::DescriptorBufferInfo transform_info{transform_buffer, 0, VK_WHOLE_SIZE};
            vk::DescriptorBufferInfo style_info{style_buffer, 0, VK_WHOLE_SIZE};
            vk::DescriptorBufferInfo clip_rects_info{clip_rects_buffer, 0, VK_WHOLE_SIZE};
            agrb::descriptor_writer writer(*_descriptor_set_layout, *agrb_ctx->descriptor_pool);
            writer.write_buffer(0, &transform_info);
            writer.write_buffer(1, &style_info);
            writer.write_buffer(2, &clip_rects_info);
            if (!writer.build(_descriptor_sets[i])) return false;
            _descriptor_buffer_transforms[i] = transform_buffer;
            _descriptor_buffer_styles[i] = style_buffer;
            _descriptor_buffer_clip_rects[i] = clip_rects_buffer;
            _descriptor_buffer_transforms_dirty[i] = false;
            _descriptor_buffer_styles_dirty[i] = false;
        }
        return true;
    }

    bool GPUPicker::create_readback_resources(agrb::device &device)
    {
        const u32 frames_in_flight = get_context().frames_in_flight;
        _readback_buffers.resize(frames_in_flight);

        for (u32 i = 0; i < frames_in_flight; ++i)
        {
            auto &buffer = _readback_buffers[i];
            buffer.instance_count = 1;
            buffer.alignment_size = AUIK_PICK_RESULT_SIZE;
            buffer.buffer_size = AUIK_PICK_RESULT_SIZE;
            buffer.buffer_usage = vk::BufferUsageFlagBits::eTransferDst;
            buffer.vma_usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            buffer.required_flags =
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
            buffer.prefered_flags = vk::MemoryPropertyFlagBits::eHostCached;

            const auto alloc_info =
                agrb::make_alloc_info(buffer.vma_usage, buffer.required_flags, buffer.prefered_flags, 1.0f);
            VmaAllocationCreateInfo host_alloc_info = alloc_info;
            host_alloc_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            if (!agrb::allocate_buffer(buffer, host_alloc_info, buffer.buffer_usage, device)) return false;
            if (!agrb::map_buffer(buffer, device)) return false;
            std::memset(buffer.mapped, 0, static_cast<size_t>(AUIK_PICK_RESULT_SIZE));
        }
        return true;
    }

    bool GPUPicker::update_descriptors(AgrbContext *ctx, u32 frame_id)
    {
        assert(!_descriptor_sets.empty());
        assert(frame_id < _descriptor_sets.size());
        assert(ctx->clip_rects);

        const auto &transforms_data = _rects[frame_id].transforms.data();
        const auto &styles_data = _rects[frame_id].styles.data();
        const auto &clip_rects_data = ctx->clip_rects[frame_id].data();
        const vk::Buffer transform_buffer = transforms_data.vk_buffer;
        const vk::Buffer style_buffer = styles_data.vk_buffer;
        const vk::Buffer clip_rects_buffer = clip_rects_data.vk_buffer;
        const bool clip_rects_reallocated = ctx->clip_rects_reallocated && ctx->clip_rects_reallocated[frame_id];
        if (!transform_buffer || !style_buffer || !clip_rects_buffer) return false;
        if (_descriptor_buffer_transforms[frame_id] == transform_buffer &&
            _descriptor_buffer_styles[frame_id] == style_buffer &&
            _descriptor_buffer_clip_rects[frame_id] == clip_rects_buffer &&
            !_descriptor_buffer_transforms_dirty[frame_id] && !_descriptor_buffer_styles_dirty[frame_id] &&
            !clip_rects_reallocated)
            return true;

        vk::DescriptorBufferInfo transform_info{transform_buffer, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo style_info{style_buffer, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo clip_rects_info{clip_rects_buffer, 0, VK_WHOLE_SIZE};
        agrb::descriptor_writer writer(*_descriptor_set_layout, *ctx->descriptor_pool);
        writer.write_buffer(0, &transform_info);
        writer.write_buffer(1, &style_info);
        writer.write_buffer(2, &clip_rects_info);
        writer.overwrite(_descriptor_sets[frame_id]);
        _descriptor_buffer_transforms[frame_id] = transform_buffer;
        _descriptor_buffer_styles[frame_id] = style_buffer;
        _descriptor_buffer_clip_rects[frame_id] = clip_rects_buffer;
        _descriptor_buffer_transforms_dirty[frame_id] = false;
        _descriptor_buffer_styles_dirty[frame_id] = false;
        return true;
    }

    void GPUPicker::destroy(agrb::device &device)
    {
        for (auto &buffer : _readback_buffers)
            if (buffer.vk_buffer) agrb::destroy_buffer(buffer, device);
        _readback_buffers.clear();

        if (attachments) agrb::destroy_framebuffer(*this, device);
        if (_rects)
        {
            const u32 frames = get_context().frames_in_flight;
            for (u32 i = 0; i < frames; ++i)
            {
                _rects[i].transforms.destroy();
                _rects[i].styles.destroy();
            }
            acul::release(_rects, frames);
            _rects = nullptr;
        }
        _descriptor_set_layout.reset();
        _descriptor_sets.clear();
        _descriptor_buffer_transforms.clear();
        _descriptor_buffer_styles.clear();
        _descriptor_buffer_clip_rects.clear();
        _descriptor_buffer_transforms_dirty.clear();
        _descriptor_buffer_styles_dirty.clear();
        _pipeline = nullptr;
        _device = nullptr;
        _depth_format = vk::Format::eUndefined;
    }

    bool GPUPicker::prepare(AgrbContext *context)
    {
        _device = &context->device;
        auto &device = context->device;
        create_render_pass(device);
        if (!create_attachments(device)) goto err;
        if (!create_descriptor_resources(device)) goto err;
        if (!create_readback_resources(device)) goto err;
        return true;
    err:
        destroy(device);
        return false;
    }

    bool GPUPicker::construct_pipeline(agrb::device &device, DrawPipeline &pipeline, bool bind_pipeline)
    {
        if (pipeline.layout) return true;
        if (bind_pipeline) _pipeline = &pipeline;
        if (!_descriptor_set_layout)
        {
            _descriptor_set_layout =
                agrb::descriptor_set_layout::builder()
                    .add_binding(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                    .add_binding(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                    .add_binding(2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eFragment)
                    .build(device);
            if (!_descriptor_set_layout) return false;
        }

        pipeline.descriptor_set_layout = _descriptor_set_layout;
        const vk::DescriptorSetLayout set_layouts[] = {pipeline.descriptor_set_layout->layout()};
        const vk::PushConstantRange push_constant{vk::ShaderStageFlagBits::eVertex, 0, sizeof(amal::vec2)};
        vk::PipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.setSetLayoutCount(1)
            .setPSetLayouts(set_layouts)
            .setPushConstantRangeCount(1)
            .setPPushConstantRanges(&push_constant);
        pipeline.layout = device.vk_device.createPipelineLayout(pipeline_layout_info, nullptr, device.loader);
        return pipeline.layout != nullptr;
    }

    bool GPUPicker::configure_pipeline(AgrbContext *ctx, agrb::graphics_pipeline_batch::artifact &artifact,
                                       DrawPipeline &pipeline)
    {
        auto *tmp = static_cast<agrb::graphics_pipeline_batch::artifact::custom_data_t<u32> *>(artifact.tmp);
        assert(tmp);

        artifact.config.load_defaults();
        artifact.config.color_blend_attachment.setBlendEnable(false);
        artifact.config.depth_stencil_info.setDepthTestEnable(true).setDepthWriteEnable(true).setDepthCompareOp(
            vk::CompareOp::eGreaterOrEqual);
        artifact.config.render_pass = get_rp();
        artifact.config.pipeline_layout = pipeline.layout;
        artifact.config.subpass = tmp->value;

        const auto &path = get_shader_library_path();
        auto &device = ctx->device;
        vk::ShaderModule shaders[2];
        auto vs = ctx->shader_cache.get_shader(AS_AUIK_PICKER_VS, shaders[0], device, path);
        if (!vs.success()) return false;
        auto fs = ctx->shader_cache.get_shader(AS_AUIK_PICKER_FS, shaders[1], device, path);
        if (!fs.success()) return false;
        agrb::prepare_base_graphics_pipeline(artifact, shaders, device);
        return true;
    }

    static inline bool check_mouse_bounds(const amal::vec2 &mouse_pos, const amal::vec2 &dimensions)
    {
        return mouse_pos.x >= 0 && mouse_pos.y >= 0 && mouse_pos.x < dimensions.x && mouse_pos.y < dimensions.y;
    }

    static void begin_render_pass(agrb::framebuffer &fb, u32 frame_id, vk::CommandBuffer cmd,
                                  const vk::DispatchLoaderDynamic &loader)
    {
        vk::RenderPassBeginInfo rp_info;
        rp_info.setRenderPass(fb.get_rp())
            .setFramebuffer(fb.get_fb(frame_id))
            .setRenderArea({{0, 0}, {1, 1}})
            .setClearValueCount(2)
            .setPClearValues(fb.clear_values);
        cmd.beginRenderPass(&rp_info, vk::SubpassContents::eInline, loader);
    }

    void GPUPicker::render(AgrbContext *ctx, vk::CommandBuffer *cmd)
    {
        auto &global_ctx = get_context();
        u32 frame_id = global_ctx.frame_id;
        auto &loader = ctx->device.loader;

        begin_render_pass(*this, frame_id, *cmd, loader);
        const auto &io = global_ctx.io;
        vk::Viewport viewport = {-static_cast<f32>(io.mouse_pos.x),
                                 -static_cast<f32>(io.mouse_pos.y),
                                 static_cast<f32>(io.display_size.x),
                                 static_cast<f32>(io.display_size.y),
                                 0.0f,
                                 1.0f};
        vk::Rect2D scissor = {{0, 0}, {1, 1}};
        cmd->setViewport(0, 1, &viewport, loader);
        cmd->setScissor(0, 1, &scissor, loader);

        auto &rects = frame_rects(frame_id);
        if (!rects.transforms.empty() && update_descriptors(ctx, frame_id))
        {
            cmd->bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline->handle, loader);
            cmd->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _pipeline->layout, 0, 1,
                                    &_descriptor_sets[frame_id], 0, nullptr, loader);
            cmd->pushConstants(_pipeline->layout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(amal::vec2),
                               &io.display_size, loader);
            cmd->draw(6, rects.transforms.size(), 0, 0, loader);
        }
        cmd->endRenderPass(loader);
        {
            vk::BufferImageCopy region{};
            region.setBufferOffset(0)
                .setBufferRowLength(0)
                .setBufferImageHeight(0)
                .setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
                .setImageOffset({0, 0, 0})
                .setImageExtent({1, 1, 1});
            cmd->copyImageToBuffer(attachments->images[frame_id].attachments[0].image,
                                   vk::ImageLayout::eTransferSrcOptimal, _readback_buffers[frame_id].vk_buffer, 1,
                                   &region, loader);
        }
    }

    void GPUPicker::pick(AgrbContext *ctx, u32 read_frame_id)
    {
        (void)ctx;
        auto *data = static_cast<const PickValue *>(_readback_buffers[read_frame_id].mapped);
        if (!data) return;
        auto &global_ctx = get_context();
        const auto prev_hover = global_ctx.hover_id;
        global_ctx.hover_id = make_element_id(data->widget_id, data->tag_id, data->element_id);
        on_hover_id_updated(prev_hover, global_ctx.hover_id);
    }

    u32 GPUPicker::push_hit_rect(const RectData &rect)
    {
        const u32 frame_id = get_context().frame_id;
        auto &rects = frame_rects(frame_id);
        const u32 id = static_cast<u32>(rects.transforms.size());
        const auto transform_result = rects.transforms.push_back(rect.bounds);
        const auto style_result =
            rects.styles.push_back(PickerStyleData{rect.id, rect.hit_depth, rect.depth, rect.clip_id, rect.flags});
        if (transform_result & agrb::vector_result_flag_bits::buffer_reallocated)
            _descriptor_buffer_transforms_dirty[frame_id] = true;
        if (style_result & agrb::vector_result_flag_bits::buffer_reallocated)
            _descriptor_buffer_styles_dirty[frame_id] = true;
        if (++rects.transform_version == 0u) ++rects.transform_version;
        if (++rects.style_version == 0u) ++rects.style_version;
        reset_dirty_pages(rects.transform_pages, rects.transforms.size(), rects.transform_version);
        reset_dirty_pages(rects.style_pages, rects.styles.size(), rects.style_version);
        return id;
    }

    void GPUPicker::update_hit_rect(u32 id, const RectData &rect)
    {
        const u32 frame_id = get_context().frame_id;
        auto &rects = frame_rects(frame_id);
        if (id >= rects.transforms.size()) return;
        const amal::rect transform = rect.bounds;
        auto &current_transform = rects.transforms[id];
        if (current_transform.offset != transform.offset || current_transform.size != transform.size)
        {
            current_transform = transform;
            if (++rects.transform_version == 0u) ++rects.transform_version;
            mark_dirty_page(rects.transform_pages, rects.transforms.size(), id, rects.transform_version);
        }
        const PickerStyleData style{rect.id, rect.hit_depth, rect.depth, rect.clip_id, rect.flags};
        auto &current_style = rects.styles[id];
        if (current_style.id != style.id || current_style.hit_depth != style.hit_depth ||
            current_style.depth != style.depth || current_style.clip_id != style.clip_id ||
            current_style.flags != style.flags)
        {
            current_style = style;
            if (++rects.style_version == 0u) ++rects.style_version;
            mark_dirty_page(rects.style_pages, rects.styles.size(), id, rects.style_version);
        }
    }

    void GPUPicker::clear_hit_rects()
    {
        auto &rects = frame_rects(get_context().frame_id);
        if (!rects.transforms.empty() && ++rects.transform_version == 0u) ++rects.transform_version;
        if (!rects.styles.empty() && ++rects.style_version == 0u) ++rects.style_version;
        rects.transforms.clear();
        rects.styles.clear();
        reset_dirty_pages(rects.transform_pages, 0u, rects.transform_version);
        reset_dirty_pages(rects.style_pages, 0u, rects.style_version);
    }

    void GPUPicker::copy_frame_data(u32 dst_frame_id, u32 src_frame_id)
    {
        if (dst_frame_id == src_frame_id || !_rects) return;
        auto &dst = frame_rects(dst_frame_id);
        auto &src = frame_rects(src_frame_id);
        const u32 count = static_cast<u32>(src.transforms.size());
        const bool transform_size_changed = dst.transforms.size() != count;
        const bool style_size_changed = dst.styles.size() != src.styles.size();
        const auto transform_result = dst.transforms.resize(count);
        const auto style_result = dst.styles.resize(src.styles.size());
        if (transform_result & agrb::vector_result_flag_bits::buffer_reallocated)
            _descriptor_buffer_transforms_dirty[dst_frame_id] = true;
        if (style_result & agrb::vector_result_flag_bits::buffer_reallocated)
            _descriptor_buffer_styles_dirty[dst_frame_id] = true;
        if (transform_size_changed || dst.transform_version != src.transform_version)
            sync_paged_buffer(dst.transforms, src.transforms, dst.transform_pages, src.transform_pages,
                              transform_size_changed);
        if (style_size_changed || dst.style_version != src.style_version)
            sync_paged_buffer(dst.styles, src.styles, dst.style_pages, src.style_pages, style_size_changed);
        dst.transform_version = src.transform_version;
        dst.style_version = src.style_version;
    }

    void update_hover_id_impl(GPUContext *gpu_context, void *sync_ctx)
    {
        auto &global_ctx = get_context();
        if (!check_mouse_bounds(global_ctx.io.mouse_pos, global_ctx.io.display_size))
        {
            const auto prev_hover = global_ctx.hover_id;
            global_ctx.hover_id = {};
            on_hover_id_updated(prev_hover, {});
            return;
        }
        auto *agrb_ctx = get_agrb_context(gpu_context);
        auto &picker = agrb_ctx->picker;
        assert(picker->attachments);
        picker->render(agrb_ctx, static_cast<vk::CommandBuffer *>(sync_ctx));
        const u32 frames_in_flight = global_ctx.frames_in_flight;
        assert(frames_in_flight > 0);
        const u32 read_frame_id = (global_ctx.frame_id + frames_in_flight - 1) % frames_in_flight;
        picker->pick(agrb_ctx, read_frame_id);
    }
} // namespace auik::detail
