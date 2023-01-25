// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

RenderpassCache::RenderpassCache(const Instance& instance, Scheduler& scheduler)
    : instance{instance}, scheduler{scheduler} {}

RenderpassCache::~RenderpassCache() {
    vk::Device device = instance.GetDevice();
    for (u32 color = 0; color <= MAX_COLOR_FORMATS; color++) {
        for (u32 depth = 0; depth <= MAX_DEPTH_FORMATS; depth++) {
            if (vk::RenderPass load_pass = cached_renderpasses[color][depth][0]; load_pass) {
                device.destroyRenderPass(load_pass);
            }

            if (vk::RenderPass clear_pass = cached_renderpasses[color][depth][1]; clear_pass) {
                device.destroyRenderPass(clear_pass);
            }
        }
    }

    device.destroyRenderPass(present_renderpass);
}

void RenderpassCache::EnterRenderpass(const RenderpassState& state) {
    const bool is_dirty = scheduler.IsStateDirty(StateFlags::Renderpass);
    if (current_state == state && !is_dirty) {
        cmd_count++;
        return;
    }

    if (current_state.renderpass) {
        ExitRenderpass();
    }

    scheduler.Record([state](vk::CommandBuffer cmdbuf) {
        const vk::RenderPassBeginInfo renderpass_begin_info = {
            .renderPass = state.renderpass,
            .framebuffer = state.framebuffer,
            .renderArea = state.render_area,
            .clearValueCount = 1,
            .pClearValues = &state.clear,
        };

        cmdbuf.beginRenderPass(renderpass_begin_info, vk::SubpassContents::eInline);
    });

    if (is_dirty) {
        scheduler.MarkStateNonDirty(StateFlags::Renderpass);
    }

    current_state = state;
}

void RenderpassCache::ExitRenderpass() {
    if (!current_state.renderpass) {
        return;
    }

    scheduler.Record([](vk::CommandBuffer cmdbuf) { cmdbuf.endRenderPass(); });
    current_state = {};

    // The Mali guide recommends flushing at the end of each major renderpass
    // Testing has shown this has a significant effect on rendering performance
    if (cmd_count > 20 && instance.IsMaliGpu()) {
        scheduler.Flush();
        cmd_count = 0;
    }
}

void RenderpassCache::CreatePresentRenderpass(vk::Format format) {
    if (!present_renderpass) {
        present_renderpass =
            CreateRenderPass(format, vk::Format::eUndefined, vk::AttachmentLoadOp::eClear,
                             vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);
    }
}

vk::RenderPass RenderpassCache::GetRenderpass(VideoCore::PixelFormat color,
                                              VideoCore::PixelFormat depth, bool is_clear) {
    const u32 color_index =
        color == VideoCore::PixelFormat::Invalid ? MAX_COLOR_FORMATS : static_cast<u32>(color);
    const u32 depth_index = depth == VideoCore::PixelFormat::Invalid
                                ? MAX_DEPTH_FORMATS
                                : (static_cast<u32>(depth) - 14);

    ASSERT_MSG(color_index <= MAX_COLOR_FORMATS && depth_index <= MAX_DEPTH_FORMATS &&
                   (color_index != MAX_COLOR_FORMATS || depth_index != MAX_DEPTH_FORMATS),
               "Invalid color index {} and/or depth_index {}", color_index, depth_index);

    vk::RenderPass& renderpass = cached_renderpasses[color_index][depth_index][is_clear];
    if (!renderpass) {
        const vk::Format color_format = instance.GetTraits(color).native;
        const vk::Format depth_format = instance.GetTraits(depth).native;
        const vk::AttachmentLoadOp load_op =
            is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
        renderpass = CreateRenderPass(color_format, depth_format, load_op,
                                      vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral);
    }

    return renderpass;
}

vk::RenderPass RenderpassCache::CreateRenderPass(vk::Format color, vk::Format depth,
                                                 vk::AttachmentLoadOp load_op,
                                                 vk::ImageLayout initial_layout,
                                                 vk::ImageLayout final_layout) const {
    u32 attachment_count = 0;
    std::array<vk::AttachmentDescription, 2> attachments;

    bool use_color = false;
    vk::AttachmentReference color_attachment_ref{};
    bool use_depth = false;
    vk::AttachmentReference depth_attachment_ref{};

    if (color != vk::Format::eUndefined) {
        attachments[attachment_count] = vk::AttachmentDescription{
            .format = color,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = initial_layout,
            .finalLayout = final_layout,
        };

        color_attachment_ref = vk::AttachmentReference{
            .attachment = attachment_count++,
            .layout = vk::ImageLayout::eGeneral,
        };

        use_color = true;
    }

    if (depth != vk::Format::eUndefined) {
        attachments[attachment_count] = vk::AttachmentDescription{
            .format = depth,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = load_op,
            .stencilStoreOp = vk::AttachmentStoreOp::eStore,
            .initialLayout = vk::ImageLayout::eGeneral,
            .finalLayout = vk::ImageLayout::eGeneral,
        };

        depth_attachment_ref = vk::AttachmentReference{
            .attachment = attachment_count++,
            .layout = vk::ImageLayout::eGeneral,
        };

        use_depth = true;
    }

    // We also require only one subpass
    const vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = use_color ? 1u : 0u,
        .pColorAttachments = &color_attachment_ref,
        .pResolveAttachments = 0,
        .pDepthStencilAttachment = use_depth ? &depth_attachment_ref : nullptr,
    };

    const vk::RenderPassCreateInfo renderpass_info = {
        .attachmentCount = attachment_count,
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 0,
        .pDependencies = nullptr,
    };

    const vk::Device device = instance.GetDevice();
    return device.createRenderPass(renderpass_info);
}

} // namespace Vulkan
