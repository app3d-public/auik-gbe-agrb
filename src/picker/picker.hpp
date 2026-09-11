#pragma once

#include <agrb/descriptors.hpp>
#include <agrb/framebuffer.hpp>
#include <agrb/pipeline.hpp>
#include <agrb/vector.hpp>
#include <amal/vector.hpp>
#include <auik/detail/events.hpp>
#include <auik/detail/fwd.hpp>
#include "../dirty_pages.hpp"

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
        bool construct_pipeline(agrb::device &device, DrawPipeline &pipeline, bool bind_pipeline = true);
        bool configure_pipeline(AgrbContext *ctx, agrb::graphics_pipeline_batch::artifact &, DrawPipeline &);

        void render(AgrbContext *ctx, vk::CommandBuffer *cmd);
        void pick(AgrbContext *ctx, u32 read_frame_id);
        u32 push_hit_rect(const RectData &rect);
        void update_hit_rect(u32 id, const RectData &rect);
        void clear_hit_rects();
        void copy_frame_data(u32 dst_frame_id, u32 src_frame_id);

    private:
        friend struct PickerCacheTest;
        inline bool is_frame_data_synced(u32 frame_id) const
        {
            if (!_rects) return true;
            const auto &frame = _rects[frame_id];
            return frame.transform_version == _master.transform_version &&
                   frame.style_version == _master.style_version;
        }
        void sync_frame_data(u32 frame_id);
        struct PickerStyleData
        {
            ElementID id{};
            f32 hit_depth = 0.0f;
            f32 depth = 0.0f;
            u16 clip_id = 0xFFFFu;
            u16 flags = 0u;
        };

        static_assert(sizeof(amal::rect) == 16u);
        static_assert(sizeof(PickerStyleData) == 24u);

        struct PickerFrameData
        {
            agrb::vector<amal::rect> transforms;
            agrb::vector<PickerStyleData> styles;
            u32 transform_version = 0u;
            u32 style_version = 0u;
            DirtyPageState transform_pages;
            DirtyPageState style_pages;
        };

        struct PickerMasterData
        {
            // Keep picker comparisons in cached CPU memory; mapped frame buffers remain write-only.
            acul::vector<amal::rect> transforms;
            acul::vector<PickerStyleData> styles;
            u32 transform_version = 0u;
            u32 style_version = 0u;
            DirtyPageState transform_pages;
            DirtyPageState style_pages;
        };

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
        PickerFrameData *_rects = nullptr;
        PickerMasterData _master;
        acul::shared_ptr<agrb::descriptor_set_layout> _descriptor_set_layout = nullptr;
        acul::vector<vk::DescriptorSet> _descriptor_sets;
        acul::vector<vk::Buffer> _descriptor_buffer_transforms;
        acul::vector<vk::Buffer> _descriptor_buffer_styles;
        acul::vector<vk::Buffer> _descriptor_buffer_clip_rects;
        acul::vector<u8> _descriptor_buffer_transforms_dirty;
        acul::vector<u8> _descriptor_buffer_styles_dirty;
        acul::vector<agrb::managed_buffer> _readback_buffers;

        void create_render_pass(agrb::device &device);
        bool create_attachments(agrb::device &device);
        bool create_descriptor_resources(agrb::device &device);
        bool create_readback_resources(agrb::device &device);
        bool update_descriptors(AgrbContext *ctx, u32 frame_id);
        inline PickerFrameData &frame_rects(u32 frame_id)
        {
            assert(_rects);
            return _rects[frame_id];
        }
    };

    void update_hover_id_impl(GPUContext *, void *);
} // namespace auik::detail
