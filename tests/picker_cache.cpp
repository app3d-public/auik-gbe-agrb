#include "../src/picker/picker.hpp"
#include <auik/detail/context.hpp>
#include <agrb/agrb.hpp>
#include <cassert>

namespace auik::detail
{
    struct PickerCacheTest
    {
        static void run(agrb::device &device)
        {
            Context context{};
            context.frames_in_flight = 2;
            g_context = &context;
            GPUPicker picker(device);
            GPUPicker::PickerFrameData frames[2];
            picker._rects = frames;
            picker._descriptor_buffer_transforms_dirty.resize(2);
            picker._descriptor_buffer_styles_dirty.resize(2);
            agrb::managed_buffer buffer{};
            buffer.buffer_usage = vk::BufferUsageFlagBits::eStorageBuffer;
            buffer.vma_usage = VMA_MEMORY_USAGE_CPU_ONLY;
            buffer.required_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
            for (auto &frame : frames)
            {
                frame.transforms.init(device, buffer);
                frame.styles.init(device, buffer);
            }
            for (u32 i = 0; i < 130; ++i)
            {
                RectData rect{};
                rect.bounds = {{static_cast<f32>(i), 0.f}, {10.f, 10.f}};
                rect.id.widget_id = i + 1;
                assert(picker.push_hit_rect(rect) == i);
            }
            picker.copy_frame_data(1, 0);

            RectData first{};
            first.bounds = {{500.f, 0.f}, {10.f, 10.f}};
            first.id.widget_id = 500;
            picker.update_hit_rect(0, first);
            context.frame_id = 1;
            RectData second = first;
            second.bounds.offset.x = 600.f;
            second.id.widget_id = 600;
            picker.update_hit_rect(1, second);
            // Updating one element must not declare its stale neighbours current.
            assert(frames[1].transforms[0].offset.x == 500.f);
            assert(frames[1].styles[0].id.widget_id == 500);
            assert(frames[1].transforms[1].offset.x == 600.f);
            assert(frames[1].transforms[129].offset.x == 129.f);

            context.frame_id = 0;
            // Equality with the CPU master is not equality with this frame slot.
            picker.update_hit_rect(1, second);
            assert(frames[0].transforms[1].offset.x == 600.f);
            assert(frames[0].styles[1].id.widget_id == 600);
            const auto transform_version = picker._master.transform_version;
            const auto style_version = picker._master.style_version;
            picker.update_hit_rect(1, second);
            assert(picker._master.transform_version == transform_version);
            assert(picker._master.style_version == style_version);

            picker.update_hit_rect(0, {}); // Hiding an area is also a partial update.
            context.frame_id = 1;
            assert(picker.push_hit_rect(second) == 130);
            assert(frames[1].transforms[0].size.x == 0.f);
            assert(frames[1].transforms[130].offset.x == 600.f);
            // Even a same-slot copy request must consult the backend's own master.
            context.frame_id = 0;
            picker.copy_frame_data(0, 0);
            assert(frames[0].transforms.size() == 131);
            assert(frames[0].transforms[130].offset.x == 600.f);
            picker.clear_hit_rects();
            context.frame_id = 1;
            assert(picker.push_hit_rect(first) == 0);
            assert(frames[1].transforms.size() == 1);
            assert(frames[1].transforms[0].offset.x == 500.f);
            picker._rects = nullptr;
            g_context = nullptr;
        }
    };
}

int main()
{
    agrb::init_library();
    agrb::device device;
    agrb::device_runtime_data runtime;
    agrb::device_create_ctx options;
    options.set_runtime_data(&runtime);
    agrb::init_device("picker_cache_test", 1, device, &options);
    assert(device.vk_device);
    auik::detail::PickerCacheTest::run(device);
    agrb::destroy_device(device);
    agrb::destroy_library();
}
