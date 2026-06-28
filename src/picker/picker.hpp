#pragma once

#include <agrb/descriptors.hpp>
#include <agrb/framebuffer.hpp>
#include <agrb/pipeline.hpp>
#include <agrb/vector.hpp>
#include <amal/vector.hpp>
#include <auik/detail/events.hpp>
#include <auik/detail/fwd.hpp>

namespace auik::detail
{
    struct AgrbContext;

    class GPUPicker final : public agrb::framebuffer
    {
    public:
        GPUPicker(agrb::device &device)
        {
            acul::vector<vk::Format> depth_candidates(
                {vk::Format::eX8D24UnormPack32, vk::Format::eD32Sfloat, vk::Format::eD16Unorm});
            _depth_format = device.find_supported_format(depth_candidates, vk::ImageTiling::eOptimal,
                                                         vk::FormatFeatureFlagBits::eDepthStencilAttachment);
        }

        bool prepare(AgrbContext *context);

        void destroy(agrb::device &device);
        bool construct_pipeline(agrb::device &device, DrawPipeline &pipeline);
        bool configure_pipeline(AgrbContext *ctx, agrb::graphics_pipeline_batch::artifact &, DrawPipeline &);

        void render(AgrbContext *ctx, vk::CommandBuffer *cmd);
        void pick(AgrbContext *ctx, u32 read_frame_id);
        u32 push_hit_rect(const RectData &rect);
        void update_hit_rect(u32 id, const RectData &rect);
        void clear_hit_rects();
        void copy_frame_data(u32 dst_frame_id, u32 src_frame_id);

    private:
        agrb::device *_device = nullptr;
        struct PickValue
        {
            u32 widget_id = 0;
            u32 tag_id = 0;
            u32 element_id = 0;
            u32 reserved = 0;
        };

        DrawPipeline *_pipeline = nullptr;
        vk::Format _depth_format = vk::Format::eUndefined;
        agrb::vector<RectData> *_rects = nullptr;
        acul::shared_ptr<agrb::descriptor_set_layout> _descriptor_set_layout = nullptr;
        acul::vector<vk::DescriptorSet> _descriptor_sets;
        acul::vector<vk::Buffer> _descriptor_buffer_instances;
        acul::vector<vk::Buffer> _descriptor_buffer_clip_rects;
        acul::vector<u8> _descriptor_buffer_instances_dirty;
        acul::vector<agrb::managed_buffer> _readback_buffers;

        void create_render_pass(agrb::device &device);
        bool create_attachments(agrb::device &device);
        bool create_descriptor_resources(agrb::device &device);
        bool create_readback_resources(agrb::device &device);
        bool update_descriptors(AgrbContext *ctx, u32 frame_id);
        inline agrb::vector<RectData> &frame_rects(u32 frame_id)
        {
            assert(_rects);
            return _rects[frame_id];
        }
    };

    void update_hover_id_impl(GPUContext *, void *);
} // namespace auik::detail
