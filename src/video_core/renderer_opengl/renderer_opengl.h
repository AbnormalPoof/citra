// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "core/hw/gpu.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_opengl/frame_dumper_opengl.h"
#include "video_core/renderer_opengl/gl_driver.h"
#include "video_core/renderer_opengl/gl_rasterizer.h"

namespace Layout {
struct FramebufferLayout;
}

namespace Memory {
class MemorySystem;
}

namespace Frontend {

struct Frame {
    u32 width{};                      /// Width of the frame (to detect resize)
    u32 height{};                     /// Height of the frame
    bool color_reloaded = false;      /// Texture attachment was recreated (ie: resized)
    OpenGL::OGLRenderbuffer color{};  /// Buffer shared between the render/present FBO
    OpenGL::OGLFramebuffer render{};  /// FBO created on the render thread
    OpenGL::OGLFramebuffer present{}; /// FBO created on the present thread
    OpenGL::OGLSync render_fence{};   /// Fence created on the render thread
    OpenGL::OGLSync present_fence{};  /// Fence created on the presentation thread
};
} // namespace Frontend

namespace OpenGL {

/**
 * Structure used for storing information about the textures for each 3DS screen
 **/
struct TextureInfo {
    OGLTexture resource;
    GLsizei width;
    GLsizei height;
    GPU::Regs::PixelFormat format;
    GLenum gl_format;
    GLenum gl_type;
};

/**
 * Structure used for storing information about the display target for each 3DS screen
 **/
struct ScreenInfo {
    GLuint display_texture;
    Common::Rectangle<f32> display_texcoords;
    TextureInfo texture;
};

class RasterizerOpenGL;

class RendererOpenGL : public VideoCore::RendererBase {
public:
    explicit RendererOpenGL(Memory::MemorySystem& memory, Frontend::EmuWindow& window,
                            Frontend::EmuWindow* secondary_window);
    ~RendererOpenGL() override;

    [[nodiscard]] VideoCore::RasterizerInterface* Rasterizer() override {
        return &rasterizer;
    }

    void SwapBuffers() override;
    void TryPresent(int timeout_ms, bool is_secondary) override;
    void PrepareVideoDumping() override;
    void CleanupVideoDumping() override;
    void Sync() override;

private:
    /**
     * Initializes the OpenGL state and creates persistent objects.
     */
    void InitOpenGLObjects();
    void ReloadSampler();
    void ReloadShader();
    void PrepareRendertarget();
    void RenderScreenshot();
    void RenderToMailbox(const Layout::FramebufferLayout& layout,
                         std::unique_ptr<Frontend::TextureMailbox>& mailbox, bool flipped);
    void ConfigureFramebufferTexture(TextureInfo& texture,
                                     const GPU::Regs::FramebufferConfig& framebuffer);

    /**
     * Draws the emulated screens to the emulator window.
     */
    void DrawScreens(const Layout::FramebufferLayout& layout, bool flipped);
    void ApplySecondLayerOpacity();
    void DrawBottomScreen(const Layout::FramebufferLayout& layout,
                          const Common::Rectangle<u32>& bottom_screen,
                          const bool stereo_single_screen);
    void DrawTopScreen(const Layout::FramebufferLayout& layout,
                       const Common::Rectangle<u32>& top_screen, const bool stereo_single_screen);

    /**
     * Draws a single texture to the emulator window.
     */
    void DrawSingleScreen(const ScreenInfo& screen_info, float x, float y, float w, float h);
    void DrawSingleScreenStereo(const ScreenInfo& screen_info_l, const ScreenInfo& screen_info_r,
                                float x, float y, float w, float h);

    /**
     * Draws a single texture to the emulator window, rotating the texture to correct for the 3DS's
     * LCD rotation.
     */
    void DrawSingleScreenRotated(const ScreenInfo& screen_info, float x, float y, float w, float h);
    void DrawSingleScreenStereoRotated(const ScreenInfo& screen_info_l,
                                       const ScreenInfo& screen_info_r, float x, float y, float w,
                                       float h);

    /**
     * Loads framebuffer from emulated memory into the active OpenGL texture.
     */
    void LoadFBToScreenInfo(const GPU::Regs::FramebufferConfig& framebuffer,
                            ScreenInfo& screen_info, bool right_eye);

    /**
     * Fills active OpenGL texture with the given RGB color. Since the color is solid, the texture
     * can be 1x1 but will stretch across whatever it's rendered on.
     */
    void LoadColorToActiveGLTexture(u8 color_r, u8 color_g, u8 color_b, const TextureInfo& texture);

private:
    Memory::MemorySystem& memory;
    Driver driver;
    OpenGLState state;
    RasterizerOpenGL rasterizer;

    // OpenGL object IDs
    OGLVertexArray vertex_array;
    OGLBuffer vertex_buffer;
    OGLProgram shader;
    OGLFramebuffer screenshot_framebuffer;
    std::array<OGLSampler, 2> present_samplers;
    u32 current_sampler = 0;

    /// Display information for top and bottom screens respectively
    std::array<ScreenInfo, 3> screen_infos;

    // Shader uniform location indices
    GLuint uniform_modelview_matrix;
    GLuint uniform_color_texture;
    GLuint uniform_color_texture_r;

    // Shader uniform for Dolphin compatibility
    GLuint uniform_i_resolution;
    GLuint uniform_o_resolution;
    GLuint uniform_layer;

    // Shader attribute input indices
    GLuint attrib_position;
    GLuint attrib_tex_coord;

    FrameDumperOpenGL frame_dumper;
};

} // namespace OpenGL
