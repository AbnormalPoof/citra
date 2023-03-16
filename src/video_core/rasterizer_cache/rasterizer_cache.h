// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <unordered_map>
#include <boost/icl/interval_map.hpp>
#include <boost/icl/interval_set.hpp>
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/renderer_opengl/gl_texture_runtime.h"
#include "video_core/texture/texture_decode.h"

namespace Memory {
class MemorySystem;
}

namespace Pica {
struct Regs;
}

namespace VideoCore {

enum class ScaleMatch {
    Exact,   // only accept same res scale
    Upscale, // only allow higher scale than params
    Ignore   // accept every scaled res
};

class RasterizerCache : NonCopyable {
public:
    using Surface = std::shared_ptr<OpenGL::Surface>;

    // Declare rasterizer interval types
    using SurfaceSet = std::set<Surface>;
    using SurfaceMap = boost::icl::interval_map<PAddr, Surface, boost::icl::partial_absorber,
                                                std::less, boost::icl::inplace_plus,
                                                boost::icl::inter_section, SurfaceInterval>;
    using SurfaceCache = boost::icl::interval_map<PAddr, SurfaceSet, boost::icl::partial_absorber,
                                                  std::less, boost::icl::inplace_plus,
                                                  boost::icl::inter_section, SurfaceInterval>;

    static_assert(std::is_same<SurfaceRegions::interval_type, SurfaceCache::interval_type>() &&
                      std::is_same<SurfaceMap::interval_type, SurfaceCache::interval_type>(),
                  "Incorrect interval types");

    using SurfaceRect_Tuple = std::tuple<Surface, Common::Rectangle<u32>>;
    using PageMap = boost::icl::interval_map<u32, int>;

    struct RenderTargets {
        Surface color_surface;
        Surface depth_surface;
    };

public:
    RasterizerCache(Memory::MemorySystem& memory, OpenGL::TextureRuntime& runtime,
                    Pica::Regs& regs);
    ~RasterizerCache();

    /// Perform hardware accelerated texture copy according to the provided configuration
    bool AccelerateTextureCopy(const GPU::Regs::DisplayTransferConfig& config);

    /// Perform hardware accelerated display transfer according to the provided configuration
    bool AccelerateDisplayTransfer(const GPU::Regs::DisplayTransferConfig& config);

    /// Perform hardware accelerated memory fill according to the provided configuration
    bool AccelerateFill(const GPU::Regs::MemoryFillConfig& config);

    /// Copy one surface's region to another
    void CopySurface(const Surface& src_surface, const Surface& dst_surface,
                     SurfaceInterval copy_interval);

    /// Load a texture from 3DS memory to OpenGL and cache it (if not already cached)
    Surface GetSurface(const SurfaceParams& params, ScaleMatch match_res_scale,
                       bool load_if_create);

    /// Attempt to find a subrect (resolution scaled) of a surface, otherwise loads a texture from
    /// 3DS memory to OpenGL and caches it (if not already cached)
    SurfaceRect_Tuple GetSurfaceSubRect(const SurfaceParams& params, ScaleMatch match_res_scale,
                                        bool load_if_create);

    /// Get a surface based on the texture configuration
    Surface GetTextureSurface(const Pica::TexturingRegs::FullTextureConfig& config);
    Surface GetTextureSurface(const Pica::Texture::TextureInfo& info, u32 max_level = 0);

    /// Get a texture cube based on the texture configuration
    const OpenGL::CachedTextureCube& GetTextureCube(const TextureCubeConfig& config);

    /// Get the color and depth surfaces based on the framebuffer configuration
    OpenGL::Framebuffer GetFramebufferSurfaces(bool using_color_fb, bool using_depth_fb);

    /// Marks the draw rectangle defined in framebuffer as invalid
    void InvalidateFramebuffer(const OpenGL::Framebuffer& framebuffer);

    /// Get a surface that matches the fill config
    Surface GetFillSurface(const GPU::Regs::MemoryFillConfig& config);

    /// Get a surface that matches a "texture copy" display transfer config
    SurfaceRect_Tuple GetTexCopySurface(const SurfaceParams& params);

    /// Write any cached resources overlapping the region back to memory (if dirty)
    void FlushRegion(PAddr addr, u32 size, Surface flush_surface = nullptr);

    /// Mark region as being invalidated by region_owner (nullptr if 3DS memory)
    void InvalidateRegion(PAddr addr, u32 size, const Surface& region_owner);

    /// Flush all cached resources tracked by this cache manager
    void FlushAll();

    /// Clear all cached resources tracked by this cache manager
    void ClearAll(bool flush);

private:
    void DuplicateSurface(const Surface& src_surface, const Surface& dest_surface);

    /// Update surface's texture for given region when necessary
    void ValidateSurface(const Surface& surface, PAddr addr, u32 size);

    /// Copies pixel data in interval from the guest VRAM to the host GPU surface
    void UploadSurface(const Surface& surface, SurfaceInterval interval);

    /// Copies pixel data in interval from the host GPU surface to the guest VRAM
    void DownloadSurface(const Surface& surface, SurfaceInterval interval);

    /// Downloads a fill surface to guest VRAM
    void DownloadFillSurface(const Surface& surface, SurfaceInterval interval);

    /// Returns false if there is a surface in the cache at the interval with the same bit-width,
    bool NoUnimplementedReinterpretations(const Surface& surface, SurfaceParams& params,
                                          const SurfaceInterval& interval);

    /// Return true if a surface with an invalid pixel format exists at the interval
    bool IntervalHasInvalidPixelFormat(SurfaceParams& params, const SurfaceInterval& interval);

    /// Attempt to find a reinterpretable surface in the cache and use it to copy for validation
    bool ValidateByReinterpretation(const Surface& surface, SurfaceParams& params,
                                    const SurfaceInterval& interval);

    /// Create a new surface
    Surface CreateSurface(const SurfaceParams& params);

    /// Register surface into the cache
    void RegisterSurface(const Surface& surface);

    /// Remove surface from the cache
    void UnregisterSurface(const Surface& surface);

    /// Increase/decrease the number of surface in pages touching the specified region
    void UpdatePagesCachedCount(PAddr addr, u32 size, int delta);

private:
    Memory::MemorySystem& memory;
    OpenGL::TextureRuntime& runtime;
    Pica::Regs& regs;
    SurfaceCache surface_cache;
    PageMap cached_pages;
    SurfaceMap dirty_regions;
    SurfaceSet remove_surfaces;
    u16 resolution_scale_factor;
    RenderTargets render_targets;
    std::unordered_map<TextureCubeConfig, OpenGL::CachedTextureCube> texture_cube_cache;
};

} // namespace VideoCore
