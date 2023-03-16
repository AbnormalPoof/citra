// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/scope_exit.h"
#include "common/settings.h"
#include "video_core/regs.h"
#include "video_core/renderer_opengl/gl_driver.h"
#include "video_core/renderer_opengl/gl_state.h"
#include "video_core/renderer_opengl/gl_texture_runtime.h"
#include "video_core/video_core.h"

namespace OpenGL {

namespace {

using VideoCore::PixelFormat;
using VideoCore::SurfaceType;

constexpr FormatTuple DEFAULT_TUPLE = {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE};

static constexpr std::array<FormatTuple, 4> DEPTH_TUPLES = {{
    {GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT}, // D16
    {},
    {GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT},   // D24
    {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8}, // D24S8
}};

static constexpr std::array<FormatTuple, 5> COLOR_TUPLES = {{
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8},     // RGBA8
    {GL_RGB8, GL_BGR, GL_UNSIGNED_BYTE},              // RGB8
    {GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1}, // RGB5A1
    {GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5},     // RGB565
    {GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4},   // RGBA4
}};

static constexpr std::array<FormatTuple, 5> COLOR_TUPLES_OES = {{
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE},            // RGBA8
    {GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE},              // RGB8
    {GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1}, // RGB5A1
    {GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5},     // RGB565
    {GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4},   // RGBA4
}};

[[nodiscard]] GLbitfield MakeBufferMask(SurfaceType type) {
    switch (type) {
    case SurfaceType::Color:
    case SurfaceType::Texture:
    case SurfaceType::Fill:
        return GL_COLOR_BUFFER_BIT;
    case SurfaceType::Depth:
        return GL_DEPTH_BUFFER_BIT;
    case SurfaceType::DepthStencil:
        return GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }
    return GL_COLOR_BUFFER_BIT;
}

[[nodiscard]] u32 FboIndex(VideoCore::SurfaceType type) {
    switch (type) {
    case VideoCore::SurfaceType::Color:
    case VideoCore::SurfaceType::Texture:
    case VideoCore::SurfaceType::Fill:
        return 0;
    case VideoCore::SurfaceType::Depth:
        return 1;
    case VideoCore::SurfaceType::DepthStencil:
        return 2;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }
    return 0;
}

[[nodiscard]] OGLTexture MakeHandle(GLenum target, u32 width, u32 height, u32 levels,
                                    const FormatTuple& tuple, std::string_view debug_name = "") {
    OGLTexture texture{};
    texture.Create();

    glBindTexture(target, texture.handle);
    glTexStorage2D(target, levels, tuple.internal_format, width, height);

    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    if (!debug_name.empty()) {
        glObjectLabel(GL_TEXTURE, texture.handle, -1, debug_name.data());
    }

    return texture;
}

} // Anonymous namespace

TextureRuntime::TextureRuntime(Driver& driver_)
    : driver{driver_}, filterer{Settings::values.texture_filter_name.GetValue(),
                                VideoCore::GetResolutionScaleFactor()} {

    for (std::size_t i = 0; i < draw_fbos.size(); ++i) {
        draw_fbos[i].Create();
        read_fbos[i].Create();
    }

    auto Register = [this](PixelFormat dest, std::unique_ptr<FormatReinterpreterBase>&& obj) {
        const u32 dst_index = static_cast<u32>(dest);
        return reinterpreters[dst_index].push_back(std::move(obj));
    };

    Register(PixelFormat::RGBA8, std::make_unique<ShaderD24S8toRGBA8>());
    Register(PixelFormat::RGB5A1, std::make_unique<RGBA4toRGB5A1>());
}

TextureRuntime::~TextureRuntime() = default;

bool TextureRuntime::ResetFilter() {
    return filterer.Reset(Settings::values.texture_filter_name.GetValue(),
                          VideoCore::GetResolutionScaleFactor());
}

VideoCore::StagingData TextureRuntime::FindStaging(u32 size, bool upload) {
    if (size > staging_buffer.size()) {
        staging_buffer.resize(size);
    }
    return VideoCore::StagingData{
        .size = size,
        .mapped = std::span{staging_buffer.data(), size},
        .buffer_offset = 0,
    };
}

const FormatTuple& TextureRuntime::GetFormatTuple(PixelFormat pixel_format) const {
    const auto type = GetFormatType(pixel_format);
    const std::size_t format_index = static_cast<std::size_t>(pixel_format);

    if (type == SurfaceType::Color) {
        ASSERT(format_index < COLOR_TUPLES.size());
        return (driver.IsOpenGLES() ? COLOR_TUPLES_OES : COLOR_TUPLES)[format_index];
    } else if (type == SurfaceType::Depth || type == SurfaceType::DepthStencil) {
        const std::size_t tuple_idx = format_index - 14;
        ASSERT(tuple_idx < DEPTH_TUPLES.size());
        return DEPTH_TUPLES[tuple_idx];
    }

    return DEFAULT_TUPLE;
}

void TextureRuntime::Recycle(const HostTextureTag tag, Allocation&& alloc) {
    texture_recycler.emplace(tag, std::move(alloc));
}

Allocation TextureRuntime::Allocate(const VideoCore::SurfaceParams& params) {
    const u32 width = params.width;
    const u32 height = params.height;
    const u32 levels = params.levels;
    const GLenum target = params.texture_type == VideoCore::TextureType::CubeMap
                              ? GL_TEXTURE_CUBE_MAP
                              : GL_TEXTURE_2D;
    const auto& tuple = GetFormatTuple(params.pixel_format);

    const HostTextureTag key = {
        .tuple = tuple,
        .type = params.texture_type,
        .width = width,
        .height = height,
        .levels = levels,
        .res_scale = params.res_scale,
    };

    if (auto it = texture_recycler.find(key); it != texture_recycler.end()) {
        Allocation alloc = std::move(it->second);
        ASSERT(alloc.res_scale == params.res_scale);
        texture_recycler.erase(it);
        return alloc;
    }

    const GLuint old_tex = OpenGLState::GetCurState().texture_units[0].texture_2d;
    glActiveTexture(GL_TEXTURE0);

    std::array<OGLTexture, 2> textures{};
    std::array<GLuint, 2> handles{};

    textures[0] = MakeHandle(target, width, height, levels, tuple, params.DebugName(false));
    handles.fill(textures[0].handle);

    if (params.res_scale != 1) {
        const u32 scaled_width = params.GetScaledWidth();
        const u32 scaled_height = params.GetScaledHeight();
        textures[1] =
            MakeHandle(target, scaled_width, scaled_height, levels, tuple, params.DebugName(true));
        handles[1] = textures[1].handle;
    }

    glBindTexture(target, old_tex);

    return Allocation{
        .textures = std::move(textures),
        .handles = std::move(handles),
        .tuple = tuple,
        .width = params.width,
        .height = params.height,
        .levels = params.levels,
        .res_scale = params.res_scale,
    };
}

bool TextureRuntime::ClearTexture(Surface& surface, const VideoCore::TextureClear& clear) {
    const auto prev_state = OpenGLState::GetCurState();

    // Setup scissor rectangle according to the clear rectangle
    OpenGLState state;
    state.scissor.enabled = true;
    state.scissor.x = clear.texture_rect.left;
    state.scissor.y = clear.texture_rect.bottom;
    state.scissor.width = clear.texture_rect.GetWidth();
    state.scissor.height = clear.texture_rect.GetHeight();
    state.draw.draw_framebuffer = draw_fbos[FboIndex(surface.type)].handle;
    state.Apply();

    switch (surface.type) {
    case SurfaceType::Color:
    case SurfaceType::Texture:
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               surface.Handle(), clear.texture_level);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0,
                               0);

        state.color_mask.red_enabled = true;
        state.color_mask.green_enabled = true;
        state.color_mask.blue_enabled = true;
        state.color_mask.alpha_enabled = true;
        state.Apply();

        glClearBufferfv(GL_COLOR, 0, clear.value.color.AsArray());
        break;
    case SurfaceType::Depth:
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                               surface.Handle(), clear.texture_level);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

        state.depth.write_mask = GL_TRUE;
        state.Apply();

        glClearBufferfv(GL_DEPTH, 0, &clear.value.depth);
        break;
    case SurfaceType::DepthStencil:
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D,
                               surface.Handle(), clear.texture_level);

        state.depth.write_mask = GL_TRUE;
        state.stencil.write_mask = -1;
        state.Apply();

        glClearBufferfi(GL_DEPTH_STENCIL, 0, clear.value.depth, clear.value.stencil);
        break;
    default:
        UNREACHABLE_MSG("Unknown surface type {}", surface.type);
        return false;
    }

    prev_state.Apply();
    return true;
}

bool TextureRuntime::CopyTextures(Surface& source, Surface& dest,
                                  const VideoCore::TextureCopy& copy) {
    const GLenum src_textarget = source.texture_type == VideoCore::TextureType::CubeMap
                                     ? GL_TEXTURE_CUBE_MAP
                                     : GL_TEXTURE_2D;
    const GLenum dst_textarget =
        dest.texture_type == VideoCore::TextureType::CubeMap ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D;
    glCopyImageSubData(source.Handle(), src_textarget, copy.src_level, copy.src_offset.x,
                       copy.src_offset.y, copy.src_layer, dest.Handle(), dst_textarget,
                       copy.dst_level, copy.dst_offset.x, copy.dst_offset.y, copy.dst_layer,
                       copy.extent.width, copy.extent.height, 1);
    return true;
}

bool TextureRuntime::CopyTextures(Surface& source, CachedTextureCube& dest,
                                  const VideoCore::TextureCopy& copy) {
    glCopyImageSubData(source.Handle(), GL_TEXTURE_2D, copy.src_level, copy.src_offset.x,
                       copy.src_offset.y, copy.src_layer, dest.texture.handle, GL_TEXTURE_CUBE_MAP,
                       copy.dst_level, copy.dst_offset.x, copy.dst_offset.y, copy.dst_layer,
                       copy.extent.width, copy.extent.height, 1);
    return true;
}

bool TextureRuntime::BlitTextures(Surface& source, Surface& dest,
                                  const VideoCore::TextureBlit& blit) {
    const auto prev_state = OpenGLState::GetCurState();

    OpenGLState state{};
    state.draw.read_framebuffer = read_fbos[FboIndex(source.type)].handle;
    state.draw.draw_framebuffer = draw_fbos[FboIndex(dest.type)].handle;
    state.Apply();

    source.Attach(GL_READ_FRAMEBUFFER, blit.src_level, blit.src_layer);
    dest.Attach(GL_DRAW_FRAMEBUFFER, blit.dst_level, blit.dst_layer);

    // TODO (wwylele): use GL_NEAREST for shadow map texture
    // Note: shadow map is treated as RGBA8 format in PICA, as well as in the rasterizer cache, but
    // doing linear intepolation componentwise would cause incorrect value. However, for a
    // well-programmed game this code path should be rarely executed for shadow map with
    // inconsistent scale.
    const GLbitfield buffer_mask = MakeBufferMask(source.type);
    const GLenum filter = buffer_mask == GL_COLOR_BUFFER_BIT ? GL_LINEAR : GL_NEAREST;
    glBlitFramebuffer(blit.src_rect.left, blit.src_rect.bottom, blit.src_rect.right,
                      blit.src_rect.top, blit.dst_rect.left, blit.dst_rect.bottom,
                      blit.dst_rect.right, blit.dst_rect.top, buffer_mask, filter);

    prev_state.Apply();
    return true;
}

void TextureRuntime::GenerateMipmaps(Surface& surface, u32 max_level) {
    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    OpenGLState state;
    state.texture_units[0].texture_2d = surface.Handle();
    state.Apply();

    glActiveTexture(GL_TEXTURE0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, max_level);

    glGenerateMipmap(GL_TEXTURE_2D);
}

const ReinterpreterList& TextureRuntime::GetPossibleReinterpretations(
    PixelFormat dest_format) const {
    return reinterpreters[static_cast<u32>(dest_format)];
}

Surface::Surface(TextureRuntime& runtime_, const VideoCore::SurfaceParams& params)
    : SurfaceBase{params}, runtime{&runtime_}, driver{&runtime_.driver} {
    if (pixel_format == PixelFormat::Invalid) {
        return;
    }

    alloc = runtime->Allocate(params);
}

Surface::~Surface() {
    if (pixel_format == PixelFormat::Invalid || !alloc) {
        return;
    }

    const HostTextureTag tag = {
        .tuple = alloc.tuple,
        .type = texture_type,
        .width = alloc.width,
        .height = alloc.height,
        .levels = alloc.levels,
        .res_scale = alloc.res_scale,
    };
    runtime->Recycle(tag, std::move(alloc));
}

void Surface::Upload(const VideoCore::BufferTextureCopy& upload,
                     const VideoCore::StagingData& staging) {
    // Ensure no bad interactions with GL_UNPACK_ALIGNMENT
    ASSERT(stride * GetBytesPerPixel(pixel_format) % 4 == 0);

    const GLuint old_tex = OpenGLState::GetCurState().texture_units[0].texture_2d;
    glPixelStorei(GL_UNPACK_ROW_LENGTH, static_cast<GLint>(stride));

    // Bind the unscaled texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, Handle(false));

    // Upload the requested rectangle of pixels
    const auto& tuple = runtime->GetFormatTuple(pixel_format);
    glTexSubImage2D(GL_TEXTURE_2D, upload.texture_level, upload.texture_rect.left,
                    upload.texture_rect.bottom, upload.texture_rect.GetWidth(),
                    upload.texture_rect.GetHeight(), tuple.format, tuple.type,
                    staging.mapped.data());

    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glBindTexture(GL_TEXTURE_2D, old_tex);

    // If texture filtering is enabled attempt to upscale with that, otherwise fallback
    // to normal blit
    const auto& filterer = runtime->GetFilterer();
    const auto unscaled_rect = upload.texture_rect;
    const auto scaled_rect = upload.texture_rect * res_scale;
    if (res_scale != 1 &&
        !filterer.Filter(Handle(false), unscaled_rect, Handle(true), scaled_rect, type)) {
        const VideoCore::TextureBlit blit = {
            .src_level = upload.texture_level,
            .dst_level = upload.texture_level,
            .src_rect = unscaled_rect,
            .dst_rect = scaled_rect,
        };
        BlitScale(blit, true);
    }

    InvalidateAllWatcher();
}

void Surface::Download(const VideoCore::BufferTextureCopy& download,
                       const VideoCore::StagingData& staging) {
    // Ensure no bad interactions with GL_PACK_ALIGNMENT
    ASSERT(stride * GetBytesPerPixel(pixel_format) % 4 == 0);

    // Scale down upscaled data before downloading it
    if (res_scale != 1) {
        const VideoCore::TextureBlit blit = {
            .src_level = download.texture_level,
            .dst_level = download.texture_level,
            .src_rect = download.texture_rect * res_scale,
            .dst_rect = download.texture_rect,
        };
        BlitScale(blit, false);
    }

    // Try to download without using an fbo. This should succeed on recent desktop drivers
    if (DownloadWithoutFbo(download, staging)) {
        return;
    }

    const u32 fbo_index = FboIndex(type);
    const auto prev_state = OpenGLState::GetCurState();

    OpenGLState state{};
    state.draw.read_framebuffer = runtime->read_fbos[fbo_index].handle;
    state.Apply();

    Attach(GL_READ_FRAMEBUFFER, download.texture_level, 0, false);

    // Read the pixel data to the staging buffer
    const auto& tuple = runtime->GetFormatTuple(pixel_format);
    glReadPixels(download.texture_rect.left, download.texture_rect.bottom,
                 download.texture_rect.GetWidth(), download.texture_rect.GetHeight(), tuple.format,
                 tuple.type, staging.mapped.data());

    // Restore previous state
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    prev_state.Apply();
}

bool Surface::DownloadWithoutFbo(const VideoCore::BufferTextureCopy& download,
                                 const VideoCore::StagingData& staging) {
    // If a partial download is requested and ARB_get_texture_sub_image (core in 4.5)
    // is not available we cannot proceed further.
    const bool is_full_download = download.texture_rect == GetRect();
    const bool has_sub_image = driver->HasArbGetTextureSubImage();
    if (driver->IsOpenGLES() || (!is_full_download && !has_sub_image)) {
        return false;
    }

    const GLuint old_tex = OpenGLState::GetCurState().texture_units[0].texture_2d;
    const auto& tuple = runtime->GetFormatTuple(pixel_format);

    glActiveTexture(GL_TEXTURE0);
    glPixelStorei(GL_PACK_ROW_LENGTH, static_cast<GLint>(stride));

    // Prefer glGetTextureSubImage in most cases since it's the fastest and most convenient option
    if (has_sub_image) {
        const GLsizei buf_size = static_cast<GLsizei>(staging.mapped.size());
        glGetTextureSubImage(Handle(false), download.texture_level, download.texture_rect.left,
                             download.texture_rect.bottom, 0, download.texture_rect.GetWidth(),
                             download.texture_rect.GetHeight(), 1, tuple.format, tuple.type,
                             buf_size, staging.mapped.data());
    } else if (is_full_download) {
        // This should only trigger for full texture downloads in oldish intel drivers
        // that only support up to 4.3
        glBindTexture(GL_TEXTURE_2D, Handle(false));
        glGetTexImage(GL_TEXTURE_2D, download.texture_level, tuple.format, tuple.type,
                      staging.mapped.data());
        glBindTexture(GL_TEXTURE_2D, old_tex);
    }

    // Restore state
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    return true;
}

void Surface::Attach(GLenum target, u32 level, u32 layer, bool scaled) {
    const GLuint handle = Handle(scaled);
    const GLenum textarget = texture_type == VideoCore::TextureType::CubeMap
                                 ? GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer
                                 : GL_TEXTURE_2D;

    switch (type) {
    case VideoCore::SurfaceType::Color:
    case VideoCore::SurfaceType::Texture:
        glFramebufferTexture2D(target, GL_COLOR_ATTACHMENT0, textarget, handle, level);
        glFramebufferTexture2D(target, GL_DEPTH_STENCIL_ATTACHMENT, textarget, 0, 0);
        break;
    case VideoCore::SurfaceType::Depth:
        glFramebufferTexture2D(target, GL_COLOR_ATTACHMENT0, textarget, 0, 0);
        glFramebufferTexture2D(target, GL_DEPTH_ATTACHMENT, textarget, handle, level);
        glFramebufferTexture2D(target, GL_STENCIL_ATTACHMENT, textarget, 0, 0);
        break;
    case VideoCore::SurfaceType::DepthStencil:
        glFramebufferTexture2D(target, GL_COLOR_ATTACHMENT0, textarget, 0, 0);
        glFramebufferTexture2D(target, GL_DEPTH_STENCIL_ATTACHMENT, textarget, handle, level);
        break;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }
}

void Surface::BlitScale(const VideoCore::TextureBlit& blit, bool up_scale) {
    const u32 fbo_index = FboIndex(type);
    const auto prev_state = OpenGLState::GetCurState();

    OpenGLState state{};
    state.draw.read_framebuffer = runtime->read_fbos[fbo_index].handle;
    state.draw.draw_framebuffer = runtime->draw_fbos[fbo_index].handle;
    state.Apply();

    Attach(GL_READ_FRAMEBUFFER, blit.src_level, blit.src_layer, !up_scale);
    Attach(GL_DRAW_FRAMEBUFFER, blit.dst_level, blit.dst_layer, up_scale);

    const GLenum buffer_mask = MakeBufferMask(type);
    const GLenum filter = buffer_mask == GL_COLOR_BUFFER_BIT ? GL_LINEAR : GL_NEAREST;
    glBlitFramebuffer(blit.src_rect.left, blit.src_rect.bottom, blit.src_rect.right,
                      blit.src_rect.top, blit.dst_rect.left, blit.dst_rect.bottom,
                      blit.dst_rect.right, blit.dst_rect.top, buffer_mask, filter);

    prev_state.Apply();
}

Framebuffer::Framebuffer(TextureRuntime& runtime, Surface* const color,
                         Surface* const depth_stencil, const Pica::Regs& regs,
                         Common::Rectangle<u32> surfaces_rect)
    : VideoCore::FramebufferBase{regs, color, depth_stencil, surfaces_rect} {

    const bool shadow_rendering = regs.framebuffer.IsShadowRendering();
    const bool has_stencil = regs.framebuffer.HasStencil();
    if (shadow_rendering && !color) {
        return; // Framebuffer won't get used
    }

    if (color) {
        attachments[0] = color->Handle();
    }
    if (depth_stencil) {
        attachments[1] = depth_stencil->Handle();
    }

    const u64 hash = Common::ComputeStructHash64(attachments);
    auto [it, new_framebuffer] = runtime.framebuffer_cache.try_emplace(hash);
    if (!new_framebuffer) {
        handle = it->second.handle;
        return;
    }

    const GLuint old_fbo = OpenGLState::GetCurState().draw.draw_framebuffer;

    OGLFramebuffer& framebuffer = it->second;
    framebuffer.Create();

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer.handle);

    if (shadow_rendering) {
        glFramebufferParameteri(GL_DRAW_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH,
                                color->width * res_scale);
        glFramebufferParameteri(GL_DRAW_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_HEIGHT,
                                color->height * res_scale);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0,
                               0);
    } else {
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               color ? color->Handle() : 0, 0);
        if (depth_stencil) {
            if (has_stencil) {
                // Attach both depth and stencil
                glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                       GL_TEXTURE_2D, depth_stencil->Handle(), 0);
            } else {
                // Attach depth
                glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                                       depth_stencil->Handle(), 0);
                // Clear stencil attachment
                glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0,
                                       0);
            }
        } else {
            // Clear both depth and stencil attachment
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D,
                                   0, 0);
        }
    }

    // Restore previous framebuffer
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, old_fbo);
}

Framebuffer::~Framebuffer() = default;

} // namespace OpenGL
