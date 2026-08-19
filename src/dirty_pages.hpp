#pragma once

#include <acul/vector.hpp>
#include <agrb/vector.hpp>
#include <amal/common.hpp>
#include <cstring>

namespace auik::detail
{
    struct DirtyPageState
    {
        acul::vector<u32> versions;
        u32 page_size = 64u;
    };

    inline u32 dirty_page_size(size_t element_count)
    {
        const size_t target = amal::max<size_t>((element_count + 15u) / 16u, 64u);
        u32 result = 64u;
        while (result < target && result <= 0x80000000u) result <<= 1u;
        return result;
    }

    inline void reset_dirty_pages(DirtyPageState &state, size_t element_count, u32 version)
    {
        state.page_size = dirty_page_size(element_count);
        const size_t page_count = (element_count + state.page_size - 1u) / state.page_size;
        state.versions.resize(page_count);
        for (auto &page_version : state.versions) page_version = version;
    }

    inline void mark_dirty_page(DirtyPageState &state, size_t element_count, size_t element_index, u32 version)
    {
        const u32 page_size = dirty_page_size(element_count);
        const size_t page_count = (element_count + page_size - 1u) / page_size;
        if (state.page_size != page_size || state.versions.size() != page_count)
            reset_dirty_pages(state, element_count, version);
        else if (element_index < element_count) state.versions[element_index / page_size] = version;
    }

    inline void mark_dirty_page_range(DirtyPageState &state, size_t element_count, size_t first, size_t count,
                                      u32 version)
    {
        if (count == 0u || first >= element_count) return;
        const size_t last = amal::min(first + count, element_count) - 1u;
        mark_dirty_page(state, element_count, first, version);
        const size_t first_page = first / state.page_size;
        const size_t last_page = last / state.page_size;
        for (size_t page = first_page; page <= last_page; ++page) state.versions[page] = version;
    }

    template <typename T>
    inline void memcpy_buffer_elements(agrb::vector<T> &dst, const agrb::vector<T> &src, size_t first, size_t count)
    {
        if (count == 0u) return;
        auto &dst_buffer = dst.data();
        const auto &src_buffer = src.data();
        assert(dst_buffer.alignment_size == src_buffer.alignment_size);
        const size_t byte_offset = static_cast<size_t>(src_buffer.alignment_size) * first;
        const size_t byte_count = static_cast<size_t>(src_buffer.alignment_size) * count;
        std::memcpy(static_cast<u8 *>(dst_buffer.mapped) + byte_offset,
                    static_cast<const u8 *>(src_buffer.mapped) + byte_offset, byte_count);
    }

    template <typename T>
    inline void sync_paged_buffer(agrb::vector<T> &dst, const agrb::vector<T> &src, DirtyPageState &dst_pages,
                                  const DirtyPageState &src_pages, bool force_full_copy)
    {
        const size_t element_count = src.size();
        if (element_count == 0u)
        {
            dst_pages = src_pages;
            return;
        }
        const bool compatible =
            dst_pages.page_size == src_pages.page_size && dst_pages.versions.size() == src_pages.versions.size();
        if (!compatible) force_full_copy = true;

        size_t changed_pages = 0u;
        if (!force_full_copy)
            for (size_t page = 0u; page < src_pages.versions.size(); ++page)
                if (dst_pages.versions[page] != src_pages.versions[page]) ++changed_pages;

        const size_t page_count = src_pages.versions.size();
        if (force_full_copy || changed_pages * 2u > page_count) memcpy_buffer_elements(dst, src, 0u, element_count);
        else
        {
            size_t page = 0u;
            while (page < page_count)
            {
                if (dst_pages.versions[page] == src_pages.versions[page])
                {
                    ++page;
                    continue;
                }
                const size_t first_page = page;
                while (page < page_count && dst_pages.versions[page] != src_pages.versions[page]) ++page;
                const size_t first = first_page * src_pages.page_size;
                const size_t count = amal::min(page * src_pages.page_size, element_count) - first;
                memcpy_buffer_elements(dst, src, first, count);
            }
        }
        dst_pages = src_pages;
    }
} // namespace auik::detail
