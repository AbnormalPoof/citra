// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <span>
#include <functional>
#include <glm/glm.hpp>
#include "common/math_util.h"
#include "video_core/renderer_vulkan/vk_buffer.h"
#include "video_core/renderer_vulkan/vk_surface_params.h"

namespace Vulkan {

/// Vulkan texture object
class Texture final : public NonCopyable {
public:
    /// Information for the creation of the target texture
    struct Info {
        u32 width, height;
        vk::Format format;
        vk::ImageType type;
        vk::ImageViewType view_type;
        vk::ImageUsageFlags usage;
        u32 multisamples = 1;
        u32 levels = 1, layers = 1;
    };

    Texture() = default;
    ~Texture();

    /// Enable move operations
    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;

    /// Create a new Vulkan texture object
    void Create(const Info& info);
    void Create(Texture& texture);
    void Adopt(const Info& info, vk::Image image);
    void Destroy();

    /// Query objects
    bool IsValid() const { return texture; }
    vk::Image GetHandle() const { return texture; }
    vk::ImageView GetView() const { return view; }
    vk::Format GetFormat() const { return info.format; }
    vk::ImageLayout GetLayout() const { return layout; }
    u32 GetSamples() const { return info.multisamples; }
    u32 GetSize() const { return image_size; }
    vk::Rect2D GetArea() const { return {{0, 0},{info.width, info.height}}; }

    /// Copies CPU side pixel data to the GPU texture buffer
    void Upload(u32 level, u32 layer, u32 row_length, vk::Rect2D region, std::span<u8> pixels);
    void Download(u32 level, u32 layer, u32 row_length, vk::Rect2D region, std::span<u8> dst);

    /// Used to transition the image to an optimal layout during transfers
    void OverrideImageLayout(vk::ImageLayout new_layout);
    void Transition(vk::CommandBuffer cmdbuffer, vk::ImageLayout new_layout);
    void Transition(vk::CommandBuffer cmdbuffer, vk::ImageLayout new_layout, u32 start_level, u32 level_count,
                    u32 start_layer, u32 layer_count);

private:
    std::vector<u8> RGBToRGBA(std::span<u8> data);
    std::vector<u64> D24S8ToD32S8(std::span<u8> data);

    std::vector<u8> RGBAToRGB(std::span<u8> data);
    std::vector<u32> D32S8ToD24S8(std::span<u8> data);

private:
    Texture::Info info{};
    vk::ImageLayout layout{};
    vk::ImageAspectFlags aspect{};
    vk::Image texture;
    vk::ImageView view;
    vk::DeviceMemory memory;
    u32 image_size{};
    bool adopted{false};
    bool is_rgb{false}, is_d24s8{false};
};

} // namespace Vulkan