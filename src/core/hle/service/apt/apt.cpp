// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include "common/archives.h"
#include "common/common_paths.h"
#include "common/file_util.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/file_sys/archive_ncch.h"
#include "core/file_sys/file_backend.h"
#include "core/hle/applets/applet.h"
#include "core/hle/kernel/mutex.h"
#include "core/hle/kernel/shared_memory.h"
#include "core/hle/romfs.h"
#include "core/hle/service/apt/applet_manager.h"
#include "core/hle/service/apt/apt.h"
#include "core/hle/service/apt/apt_a.h"
#include "core/hle/service/apt/apt_s.h"
#include "core/hle/service/apt/apt_u.h"
#include "core/hle/service/apt/bcfnt/bcfnt.h"
#include "core/hle/service/apt/ns_s.h"
#include "core/hle/service/cfg/cfg.h"
#include "core/hle/service/fs/archive.h"
#include "core/hle/service/ptm/ptm.h"
#include "core/hle/service/service.h"
#include "core/hw/aes/ccm.h"
#include "core/hw/aes/key.h"

SERVICE_CONSTRUCT_IMPL(Service::APT::Module)

namespace Service::APT {

template <class Archive>
void Module::serialize(Archive& ar, const unsigned int file_version) {
    ar& shared_font_mem;
    ar& shared_font_loaded;
    ar& shared_font_relocated;
    ar& cpu_percent;
    ar& unknown_ns_state_field;
    ar& screen_capture_post_permission;
    ar& applet_manager;
    if (file_version > 0) {
        ar& wireless_reboot_info;
    }
}

SERIALIZE_IMPL(Module)

Module::NSInterface::NSInterface(std::shared_ptr<Module> apt, const char* name, u32 max_session)
    : ServiceFramework(name, max_session), apt(std::move(apt)) {}

Module::NSInterface::~NSInterface() = default;

std::shared_ptr<Module> Module::NSInterface::GetModule() const {
    return apt;
}

void Module::NSInterface::SetWirelessRebootInfo(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x06, 1, 2); // 0x00060042
    const auto size = rp.Pop<u32>();
    const auto buffer = rp.PopStaticBuffer();

    apt->wireless_reboot_info = std::move(buffer);

    auto rb = rp.MakeBuilder(1, 0);
    rb.Push(RESULT_SUCCESS);

    LOG_WARNING(Service_APT, "called size={}", size);
}

void Module::NSInterface::ShutdownAsync(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0xE, 0, 0); // 0xE0000

    LOG_INFO(Service_APT, "called");

    apt->system.RequestShutdown();

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(RESULT_SUCCESS);
}

void Module::NSInterface::RebootSystem(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x10, 6, 0); // 0x100180
    const auto launch_title = rp.Pop<u8>() != 0;
    const auto title_id = rp.Pop<u64>();
    const auto media_type = static_cast<FS::MediaType>(rp.Pop<u8>());
    rp.Skip(1, false); // Skip padding
    const auto mem_type = rp.Pop<u8>();

    LOG_WARNING(Service_APT,
                "called launch_title={}, title_id={:016X}, media_type={:02X}, mem_type={:02X}",
                launch_title, title_id, media_type, mem_type);

    // TODO: Implement loading a specific title.
    apt->system.RequestReset();

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(RESULT_SUCCESS);
}

void Module::NSInterface::RebootSystemClean(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x16, 0, 0); // 0x160000

    LOG_INFO(Service_APT, "called");

    apt->system.RequestReset();

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(RESULT_SUCCESS);
}

void Module::APTInterface::Initialize(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x2, 2, 0); // 0x20080
    const auto app_id = rp.PopEnum<AppletId>();
    const auto attributes = rp.Pop<u32>();

    LOG_DEBUG(Service_APT, "called app_id={:#010X}, attributes={:#010X}", app_id, attributes);

    auto result = apt->applet_manager->Initialize(app_id, attributes);
    if (result.Failed()) {
        IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
        rb.Push(result.Code());
    } else {
        IPC::RequestBuilder rb = rp.MakeBuilder(1, 3);
        rb.Push(RESULT_SUCCESS);
        rb.PushCopyObjects(result->notification_event, result->parameter_event);
    }
}

static u32 DecompressLZ11(const u8* in, u8* out) {
    u32_le decompressed_size;
    memcpy(&decompressed_size, in, sizeof(u32));
    in += 4;

    u8 type = decompressed_size & 0xFF;
    ASSERT(type == 0x11);
    decompressed_size >>= 8;

    u32 current_out_size = 0;
    u8 flags = 0, mask = 1;
    while (current_out_size < decompressed_size) {
        if (mask == 1) {
            flags = *(in++);
            mask = 0x80;
        } else {
            mask >>= 1;
        }

        if (flags & mask) {
            u8 byte1 = *(in++);
            u32 length = byte1 >> 4;
            u32 offset;
            if (length == 0) {
                u8 byte2 = *(in++);
                u8 byte3 = *(in++);
                length = (((byte1 & 0x0F) << 4) | (byte2 >> 4)) + 0x11;
                offset = (((byte2 & 0x0F) << 8) | byte3) + 0x1;
            } else if (length == 1) {
                u8 byte2 = *(in++);
                u8 byte3 = *(in++);
                u8 byte4 = *(in++);
                length = (((byte1 & 0x0F) << 12) | (byte2 << 4) | (byte3 >> 4)) + 0x111;
                offset = (((byte3 & 0x0F) << 8) | byte4) + 0x1;
            } else {
                u8 byte2 = *(in++);
                length = (byte1 >> 4) + 0x1;
                offset = (((byte1 & 0x0F) << 8) | byte2) + 0x1;
            }

            for (u32 i = 0; i < length; i++) {
                *out = *(out - offset);
                ++out;
            }

            current_out_size += length;
        } else {
            *(out++) = *(in++);
            current_out_size++;
        }
    }
    return decompressed_size;
}

bool Module::LoadSharedFont() {
    auto cfg = Service::CFG::GetModule(system);
    ASSERT_MSG(cfg, "CFG Module missing!");

    u8 font_region_code;
    switch (cfg->GetRegionValue()) {
    case 4: // CHN
        font_region_code = 2;
        break;
    case 5: // KOR
        font_region_code = 3;
        break;
    case 6: // TWN
        font_region_code = 4;
        break;
    default: // JPN/EUR/USA
        font_region_code = 1;
        break;
    }

    const u64_le shared_font_archive_id_low = 0x0004009b00014002 | ((font_region_code - 1) << 8);

    FileSys::NCCHArchive archive(shared_font_archive_id_low, Service::FS::MediaType::NAND);
    // 20-byte all zero path for opening RomFS
    const FileSys::Path file_path(std::vector<u8>(20, 0));
    FileSys::Mode open_mode = {};
    open_mode.read_flag.Assign(1);
    auto file_result = archive.OpenFile(file_path, open_mode);
    if (file_result.Failed())
        return false;

    auto romfs = std::move(file_result).Unwrap();
    std::vector<u8> romfs_buffer(romfs->GetSize());
    romfs->Read(0, romfs_buffer.size(), romfs_buffer.data());
    romfs->Close();

    const char16_t* file_name[4] = {u"cbf_std.bcfnt.lz", u"cbf_zh-Hans-CN.bcfnt.lz",
                                    u"cbf_ko-Hang-KR.bcfnt.lz", u"cbf_zh-Hant-TW.bcfnt.lz"};
    const RomFS::RomFSFile font_file =
        RomFS::GetFile(romfs_buffer.data(), {file_name[font_region_code - 1]});
    if (font_file.Data() == nullptr)
        return false;

    struct {
        u32_le status;
        u32_le region;
        u32_le decompressed_size;
        INSERT_PADDING_WORDS(0x1D);
    } shared_font_header{};
    static_assert(sizeof(shared_font_header) == 0x80, "shared_font_header has incorrect size");

    shared_font_header.status = 2; // successfully loaded
    shared_font_header.region = font_region_code;
    shared_font_header.decompressed_size =
        DecompressLZ11(font_file.Data(), shared_font_mem->GetPointer(0x80));
    std::memcpy(shared_font_mem->GetPointer(), &shared_font_header, sizeof(shared_font_header));
    *shared_font_mem->GetPointer(0x83) = 'U'; // Change the magic from "CFNT" to "CFNU"

    return true;
}

bool Module::LoadLegacySharedFont() {
    // This is the legacy method to load shared font.
    // The expected format is a decrypted, uncompressed BCFNT file with the 0x80 byte header
    // generated by the APT:U service. The best way to get is by dumping it from RAM. We've provided
    // a homebrew app to do this: https://github.com/citra-emu/3dsutils. Put the resulting file
    // "shared_font.bin" in the Citra "sysdata" directory.
    std::string filepath = FileUtil::GetUserPath(FileUtil::UserPath::SysDataDir) + SHARED_FONT;

    FileUtil::CreateFullPath(filepath); // Create path if not already created
    FileUtil::IOFile file(filepath, "rb");
    if (file.IsOpen()) {
        file.ReadBytes(shared_font_mem->GetPointer(), file.GetSize());
        return true;
    }

    return false;
}

void Module::APTInterface::GetSharedFont(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x44, 0, 0); // 0x00440000
    IPC::RequestBuilder rb = rp.MakeBuilder(2, 2);

    // Log in telemetry if the game uses the shared font
    apt->system.TelemetrySession().AddField(Common::Telemetry::FieldType::Session,
                                            "RequiresSharedFont", true);

    if (!apt->shared_font_loaded) {
        // On real 3DS, font loading happens on booting. However, we load it on demand to coordinate
        // with CFG region auto configuration, which happens later than APT initialization.
        if (apt->LoadSharedFont()) {
            apt->shared_font_loaded = true;
        } else if (apt->LoadLegacySharedFont()) {
            LOG_WARNING(Service_APT, "Loaded shared font by legacy method");
            apt->shared_font_loaded = true;
        } else {
            LOG_ERROR(Service_APT, "shared font file missing - go dump it from your 3ds");
            rb.Push<u32>(-1); // TODO: Find the right error code
            rb.Push<u32>(0);
            rb.PushCopyObjects<Kernel::Object>(nullptr);
            apt->system.SetStatus(Core::System::ResultStatus::ErrorSystemFiles, "Shared fonts");
            return;
        }
    }

    // The shared font has to be relocated to the new address before being passed to the
    // application.

    // Note: the target address is still in the old linear heap region even on new firmware
    // versions. This exception is made for shared font to resolve the following compatibility
    // issue:
    // The linear heap region changes depending on the kernel version marked in application's
    // exheader (not the actual version the application is running on). If an application with old
    // kernel version and an applet with new kernel version run at the same time, and they both use
    // shared font, different linear heap region would have required shared font to relocate
    // according to two different addresses at the same time, which is impossible.
    VAddr target_address = static_cast<VAddr>(apt->shared_font_mem->GetLinearHeapPhysicalOffset()) +
                           Memory::LINEAR_HEAP_VADDR;
    if (!apt->shared_font_relocated) {
        BCFNT::RelocateSharedFont(apt->shared_font_mem, target_address);
        apt->shared_font_relocated = true;
    }

    rb.Push(RESULT_SUCCESS); // No error
    // Since the SharedMemory interface doesn't provide the address at which the memory was
    // allocated, the real APT service calculates this address by scanning the entire address space
    // (using svcQueryMemory) and searches for an allocation of the same size as the Shared Font.
    rb.Push(target_address);
    rb.PushCopyObjects(apt->shared_font_mem);
}

void Module::APTInterface::GetWirelessRebootInfo(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x45, 1, 0); // 0x00450040
    const auto size = rp.Pop<u32>();

    LOG_WARNING(Service_APT, "called size={:08X}", size);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 2);
    rb.Push(RESULT_SUCCESS);
    rb.PushStaticBuffer(apt->wireless_reboot_info, 0);
}

void Module::APTInterface::NotifyToWait(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x43, 1, 0); // 0x430040
    const auto app_id = rp.Pop<u32>();

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(RESULT_SUCCESS); // No error

    LOG_WARNING(Service_APT, "(STUBBED) app_id={}", app_id);
}

void Module::APTInterface::GetLockHandle(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x1, 1, 0); // 0x10040

    // Bits [0:2] are the applet type (System, Library, etc)
    // Bit 5 tells the application that there's a pending APT parameter,
    // this will cause the app to wait until parameter_event is signaled.
    const auto attributes = rp.Pop<u32>();

    LOG_DEBUG(Service_APT, "called applet_attributes={:#010X}", attributes);

    auto result = apt->applet_manager->GetLockHandle(attributes);
    if (result.Failed()) {
        IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
        rb.Push(result.Code());
    } else {
        IPC::RequestBuilder rb = rp.MakeBuilder(3, 2);
        rb.Push(RESULT_SUCCESS);
        rb.PushRaw(result->corrected_attributes);
        rb.Push<u32>(result->state);
        rb.PushCopyObjects(result->lock);
    }
}

void Module::APTInterface::Enable(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x3, 1, 0); // 0x30040
    const auto attributes = rp.Pop<u32>();

    LOG_DEBUG(Service_APT, "called attributes={:#010X}", attributes);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->Enable(attributes));
}

void Module::APTInterface::GetAppletManInfo(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x5, 1, 0); // 0x50040
    auto applet_pos = rp.PopEnum<AppletPos>();

    LOG_DEBUG(Service_APT, "called, applet_pos={:08X}", applet_pos);

    auto info = apt->applet_manager->GetAppletManInfo(applet_pos);
    if (info.Failed()) {
        IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
        rb.Push(info.Code());
    } else {
        IPC::RequestBuilder rb = rp.MakeBuilder(5, 0);
        rb.Push(RESULT_SUCCESS);
        rb.PushEnum(info->active_applet_pos);
        rb.PushEnum(info->requested_applet_id);
        rb.PushEnum(info->home_menu_applet_id);
        rb.PushEnum(info->active_applet_id);
    }
}

void Module::APTInterface::IsRegistered(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x9, 1, 0); // 0x90040
    const auto app_id = rp.PopEnum<AppletId>();

    IPC::RequestBuilder rb = rp.MakeBuilder(2, 0);
    rb.Push(RESULT_SUCCESS); // No error
    rb.Push(apt->applet_manager->IsRegistered(app_id));

    LOG_DEBUG(Service_APT, "called app_id={:#010X}", app_id);
}

void Module::APTInterface::InquireNotification(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0xB, 1, 0); // 0xB0040
    const auto app_id = rp.PopEnum<AppletId>();

    LOG_DEBUG(Service_APT, "called app_id={:#010X}", app_id);

    auto notification = apt->applet_manager->InquireNotification(app_id);
    if (notification.Failed()) {
        IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
        rb.Push(notification.Code());
    } else {
        IPC::RequestBuilder rb = rp.MakeBuilder(2, 0);
        rb.Push(RESULT_SUCCESS);
        rb.Push(static_cast<u32>(notification.Unwrap()));
    }
}

void Module::APTInterface::SendParameter(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0xC, 4, 4); // 0xC0104
    const auto src_app_id = rp.PopEnum<AppletId>();
    const auto dst_app_id = rp.PopEnum<AppletId>();
    const auto signal_type = rp.PopEnum<SignalType>();
    const auto buffer_size = rp.Pop<u32>();
    const auto object = rp.PopGenericObject();
    const auto buffer = rp.PopStaticBuffer();

    LOG_DEBUG(Service_APT,
              "called src_app_id={:#010X}, dst_app_id={:#010X}, signal_type={:#010X},"
              "buffer_size={:#010X}",
              src_app_id, dst_app_id, signal_type, buffer_size);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->SendParameter({
        .sender_id = src_app_id,
        .destination_id = dst_app_id,
        .signal = signal_type,
        .object = object,
        .buffer = buffer,
    }));
}

void Module::APTInterface::ReceiveParameter(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0xD, 2, 0); // 0xD0080
    const auto app_id = rp.PopEnum<AppletId>();
    const auto buffer_size = rp.Pop<u32>();

    LOG_DEBUG(Service_APT, "called app_id={:#010X}, buffer_size={:#010X}", app_id, buffer_size);

    auto next_parameter = apt->applet_manager->ReceiveParameter(app_id);
    if (next_parameter.Failed()) {
        IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
        rb.Push(next_parameter.Code());
    } else {
        const auto size = std::min(static_cast<u32>(next_parameter->buffer.size()), buffer_size);
        next_parameter->buffer.resize(
            buffer_size); // APT always push a buffer with the maximum size

        IPC::RequestBuilder rb = rp.MakeBuilder(4, 4);
        rb.Push(RESULT_SUCCESS); // No error
        rb.PushEnum(next_parameter->sender_id);
        rb.PushEnum(next_parameter->signal); // Signal type
        rb.Push(size);                       // Parameter buffer size
        rb.PushMoveObjects(next_parameter->object);
        rb.PushStaticBuffer(std::move(next_parameter->buffer), 0);
    }
}

void Module::APTInterface::GlanceParameter(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0xE, 2, 0); // 0xE0080
    const auto app_id = rp.PopEnum<AppletId>();
    const u32 buffer_size = rp.Pop<u32>();

    LOG_DEBUG(Service_APT, "called app_id={:#010X}, buffer_size={:#010X}", app_id, buffer_size);

    auto next_parameter = apt->applet_manager->GlanceParameter(app_id);
    if (next_parameter.Failed()) {
        IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
        rb.Push(next_parameter.Code());
    } else {
        const auto size = std::min(static_cast<u32>(next_parameter->buffer.size()), buffer_size);
        next_parameter->buffer.resize(
            buffer_size); // APT always push a buffer with the maximum size

        IPC::RequestBuilder rb = rp.MakeBuilder(4, 4);
        rb.Push(RESULT_SUCCESS); // No error
        rb.PushEnum(next_parameter->sender_id);
        rb.PushEnum(next_parameter->signal); // Signal type
        rb.Push(size);                       // Parameter buffer size
        rb.PushMoveObjects(next_parameter->object);
        rb.PushStaticBuffer(std::move(next_parameter->buffer), 0);
    }
}

void Module::APTInterface::CancelParameter(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0xF, 4, 0); // 0xF0100
    const auto check_sender = rp.Pop<bool>();
    const auto sender_appid = rp.PopEnum<AppletId>();
    const auto check_receiver = rp.Pop<bool>();
    const auto receiver_appid = rp.PopEnum<AppletId>();

    LOG_DEBUG(
        Service_APT,
        "called check_sender={}, sender_appid={:#010X}, check_receiver={}, receiver_appid={:#010X}",
        check_sender, sender_appid, check_receiver, receiver_appid);

    IPC::RequestBuilder rb = rp.MakeBuilder(2, 0);
    rb.Push(RESULT_SUCCESS); // No error
    rb.Push(apt->applet_manager->CancelParameter(check_sender, sender_appid, check_receiver,
                                                 receiver_appid));
}

void Module::APTInterface::PrepareToDoApplicationJump(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x31, 4, 0); // 0x00310100
    auto flags = rp.PopEnum<ApplicationJumpFlags>();
    u64 title_id = rp.Pop<u64>();
    u8 media_type = rp.Pop<u8>();

    LOG_INFO(Service_APT, "called title_id={:016X}, media_type={:#01X}, flags={:#08X}", title_id,
             media_type, flags);

    ResultCode result = apt->applet_manager->PrepareToDoApplicationJump(
        title_id, static_cast<FS::MediaType>(media_type), flags);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(result);
}

void Module::APTInterface::DoApplicationJump(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x32, 2, 4); // 0x00320084
    const auto param_size = rp.Pop<u32>();
    const auto hmac_size = rp.Pop<u32>();
    const auto param = rp.PopStaticBuffer();
    const auto hmac = rp.PopStaticBuffer();

    LOG_INFO(Service_APT, "called param_size={:08X}, hmac_size={:08X}", param_size, hmac_size);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->DoApplicationJump(DeliverArg{param, hmac}));
}

void Module::APTInterface::GetProgramIdOnApplicationJump(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x33, 0, 0); // 0x00330000

    LOG_DEBUG(Service_APT, "called");

    const auto parameters = apt->applet_manager->GetApplicationJumpParameters();

    IPC::RequestBuilder rb = rp.MakeBuilder(7, 0);
    rb.Push(RESULT_SUCCESS);
    rb.Push<u64>(parameters.current_title_id);
    rb.Push(static_cast<u8>(parameters.current_media_type));
    rb.Push<u64>(parameters.next_title_id);
    rb.Push(static_cast<u8>(parameters.next_media_type));
}

void Module::APTInterface::ReceiveDeliverArg(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x35, 2, 0); // 0x00350080
    const auto param_size = rp.Pop<u32>();
    const auto hmac_size = rp.Pop<u32>();

    LOG_DEBUG(Service_APT, "called param_size={:08X}, hmac_size={:08X}", param_size, hmac_size);

    auto arg = apt->applet_manager->ReceiveDeliverArg().value_or(DeliverArg{});
    arg.param.resize(param_size);
    arg.hmac.resize(std::min<std::size_t>(hmac_size, 0x20));

    IPC::RequestBuilder rb = rp.MakeBuilder(4, 4);
    rb.Push(RESULT_SUCCESS);
    rb.Push(arg.source_program_id);
    rb.Push<u8>(1);
    rb.PushStaticBuffer(std::move(arg.param), 0);
    rb.PushStaticBuffer(std::move(arg.hmac), 1);
}

void Module::APTInterface::PrepareToStartApplication(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x15, 5, 0); // 0x00150140
    const auto title_id = rp.Pop<u64>();
    const auto media_type = static_cast<FS::MediaType>(rp.Pop<u8>());
    rp.Skip(1, false); // Padding
    const auto flags = rp.Pop<u32>();

    LOG_INFO(Service_APT, "called title_id={:#010X} media_type={} flags={:#010X}", title_id,
             media_type, flags);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->PrepareToStartApplication(title_id, media_type));
}

void Module::APTInterface::StartApplication(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x1B, 3, 4); // 0x001B00C4
    const auto parameter_size = rp.Pop<u32>();
    const auto hmac_size = rp.Pop<u32>();
    const auto paused = rp.Pop<bool>();
    const auto parameter = rp.PopStaticBuffer();
    const auto hmac = rp.PopStaticBuffer();

    LOG_INFO(Service_APT, "called parameter_size={:#010X}, hmac_size={:#010X}, paused={}",
             parameter_size, hmac_size, paused);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->StartApplication(parameter, hmac, paused));
}

void Module::APTInterface::WakeupApplication(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x1C, 0, 0); // 0x001C0000

    LOG_DEBUG(Service_APT, "called");

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->WakeupApplication());
}

void Module::APTInterface::CancelApplication(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x1D, 0, 0); // 0x001D0000

    LOG_DEBUG(Service_APT, "called");

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->CancelApplication());
}

void Module::APTInterface::AppletUtility(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x4B, 3, 2); // 0x004B00C2

    // These are from 3dbrew - I'm not really sure what they're used for.
    const auto utility_command = rp.Pop<u32>();
    const auto input_size = rp.Pop<u32>();
    const auto output_size = rp.Pop<u32>();
    [[maybe_unused]] const auto input = rp.PopStaticBuffer();

    LOG_WARNING(Service_APT,
                "(STUBBED) called command={:#010X}, input_size={:#010X}, output_size={:#010X}",
                utility_command, input_size, output_size);

    std::vector<u8> out(output_size);
    if (utility_command == 0x6 && output_size > 0) {
        // Command 0x6 (TryLockTransition) expects a boolean return value indicating
        // whether the attempt succeeded. Since we don't implement any of the transition
        // locking stuff yet, fake a success result to avoid app crashes.
        out[0] = true;
    }

    IPC::RequestBuilder rb = rp.MakeBuilder(2, 2);
    rb.Push(RESULT_SUCCESS); // No error
    rb.Push(RESULT_SUCCESS); // Utility function result
    rb.PushStaticBuffer(out, 0);
}

void Module::APTInterface::SetAppCpuTimeLimit(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x4F, 2, 0); // 0x4F0080
    const auto must_be_one = rp.Pop<u32>();
    const auto value = rp.Pop<u32>();

    LOG_WARNING(Service_APT, "(STUBBED) called, must_be_one={}, value={}", must_be_one, value);
    if (must_be_one != 1) {
        LOG_ERROR(Service_APT, "This value should be one, but is actually {}!", must_be_one);
    }

    apt->cpu_percent = value;

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(RESULT_SUCCESS); // No error
}

void Module::APTInterface::GetAppCpuTimeLimit(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x50, 1, 0); // 0x500040
    const auto must_be_one = rp.Pop<u32>();

    LOG_WARNING(Service_APT, "(STUBBED) called, must_be_one={}", must_be_one);
    if (must_be_one != 1) {
        LOG_ERROR(Service_APT, "This value should be one, but is actually {}!", must_be_one);
    }

    IPC::RequestBuilder rb = rp.MakeBuilder(2, 0);
    rb.Push(RESULT_SUCCESS); // No error
    rb.Push(apt->cpu_percent);
}

void Module::APTInterface::PrepareToStartLibraryApplet(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x18, 1, 0); // 0x180040
    const auto applet_id = rp.PopEnum<AppletId>();

    LOG_DEBUG(Service_APT, "called, applet_id={:08X}", applet_id);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->PrepareToStartLibraryApplet(applet_id));
}

void Module::APTInterface::PrepareToStartSystemApplet(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x19, 1, 0); // 0x190040
    const auto applet_id = rp.PopEnum<AppletId>();

    LOG_DEBUG(Service_APT, "called, applet_id={:08X}", applet_id);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->PrepareToStartSystemApplet(applet_id));
}

void Module::APTInterface::PrepareToStartNewestHomeMenu(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x1A, 0, 0); // 0x1A0000
    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);

    LOG_DEBUG(Service_APT, "called");

    // TODO(Subv): This command can only be called by a System Applet (return 0xC8A0CC04 otherwise).

    // This command must return an error when called, otherwise the Home Menu will try to reboot the
    // system.
    rb.Push(ResultCode(ErrorDescription::AlreadyExists, ErrorModule::Applet,
                       ErrorSummary::InvalidState, ErrorLevel::Status));
}

void Module::APTInterface::PreloadLibraryApplet(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x16, 1, 0); // 0x160040
    const auto applet_id = rp.PopEnum<AppletId>();

    LOG_DEBUG(Service_APT, "called, applet_id={:08X}", applet_id);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->PreloadLibraryApplet(applet_id));
}

void Module::APTInterface::FinishPreloadingLibraryApplet(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x17, 1, 0); // 0x00170040
    const auto applet_id = rp.PopEnum<AppletId>();

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->FinishPreloadingLibraryApplet(applet_id));

    LOG_WARNING(Service_APT, "(STUBBED) called, applet_id={:#05X}", applet_id);
}

void Module::APTInterface::StartLibraryApplet(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x1E, 2, 4); // 0x1E0084
    const auto applet_id = rp.PopEnum<AppletId>();
    const auto buffer_size = rp.Pop<u32>();
    const auto object = rp.PopGenericObject();
    const auto buffer = rp.PopStaticBuffer();

    LOG_DEBUG(Service_APT, "called, applet_id={:08X}, size={:08X}", applet_id, buffer_size);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->StartLibraryApplet(applet_id, object, buffer));
}

void Module::APTInterface::StartSystemApplet(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x1F, 2, 4); // 0x1F0084
    const auto applet_id = rp.PopEnum<AppletId>();
    const auto buffer_size = rp.Pop<u32>();
    const auto object = rp.PopGenericObject();
    const auto buffer = rp.PopStaticBuffer();

    LOG_DEBUG(Service_APT, "called, applet_id={:08X}, size={:08X}", applet_id, buffer_size);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->StartSystemApplet(applet_id, object, buffer));
}

void Module::APTInterface::OrderToCloseApplication(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x21, 0, 0);

    LOG_DEBUG(Service_APT, "called");

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->OrderToCloseApplication());
}

void Module::APTInterface::PrepareToCloseApplication(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x22, 1, 0);
    const auto return_to_sys = rp.Pop<u8>() != 0;

    LOG_DEBUG(Service_APT, "called return_to_sys={}", return_to_sys);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->PrepareToCloseApplication(return_to_sys));
}

void Module::APTInterface::CloseApplication(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x27, 1, 4);
    const auto parameter_size = rp.Pop<u32>();
    const auto object = rp.PopGenericObject();
    const auto buffer = rp.PopStaticBuffer();

    LOG_DEBUG(Service_APT, "called size={}", parameter_size);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->CloseApplication(object, buffer));
}

void Module::APTInterface::CancelLibraryApplet(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x3B, 1, 0); // 0x003B0040
    const auto app_exiting = rp.Pop<bool>();

    LOG_DEBUG(Service_APT, "called app_exiting={}", app_exiting);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->CancelLibraryApplet(app_exiting));
}

void Module::APTInterface::PrepareToCloseLibraryApplet(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x25, 3, 0); // 0x002500C0
    const auto not_pause = rp.Pop<bool>();
    const auto exiting = rp.Pop<bool>();
    const auto jump_to_home = rp.Pop<bool>();

    LOG_DEBUG(Service_APT, "called not_pause={} exiting={} jump_to_home={}", not_pause, exiting,
              jump_to_home);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->PrepareToCloseLibraryApplet(not_pause, exiting, jump_to_home));
}

void Module::APTInterface::PrepareToCloseSystemApplet(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x26, 0, 0); // 0x260000

    LOG_DEBUG(Service_APT, "called");

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->PrepareToCloseSystemApplet());
}

void Module::APTInterface::CloseLibraryApplet(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x28, 1, 4); // 0x00280044
    const auto parameter_size = rp.Pop<u32>();
    const auto object = rp.PopGenericObject();
    const auto buffer = rp.PopStaticBuffer();

    LOG_DEBUG(Service_APT, "called size={}", parameter_size);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->CloseLibraryApplet(object, buffer));
}

void Module::APTInterface::CloseSystemApplet(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x29, 1, 4); // 0x00280044
    const auto parameter_size = rp.Pop<u32>();
    const auto object = rp.PopGenericObject();
    const auto buffer = rp.PopStaticBuffer();

    LOG_DEBUG(Service_APT, "called size={}", parameter_size);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->CloseSystemApplet(object, buffer));
}

void Module::APTInterface::OrderToCloseSystemApplet(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x2A, 0, 0); // 0x2A0000

    LOG_DEBUG(Service_APT, "called");

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->OrderToCloseSystemApplet());
}

void Module::APTInterface::PrepareToJumpToHomeMenu(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x2B, 0, 0); // 0x2B0000

    LOG_DEBUG(Service_APT, "called");

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->PrepareToJumpToHomeMenu());
}

void Module::APTInterface::JumpToHomeMenu(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x2C, 1, 4); // 0x2C0044
    const auto parameter_size = rp.Pop<u32>();
    const auto object = rp.PopGenericObject();
    const auto buffer = rp.PopStaticBuffer();

    LOG_DEBUG(Service_APT, "called size={}", parameter_size);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->JumpToHomeMenu(object, buffer));
}

void Module::APTInterface::PrepareToLeaveHomeMenu(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x2D, 0, 0); // 0x2D0000

    LOG_DEBUG(Service_APT, "called");

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->PrepareToLeaveHomeMenu());
}

void Module::APTInterface::LeaveHomeMenu(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x2E, 1, 4); // 0x2E0044
    const auto parameter_size = rp.Pop<u32>();
    const auto object = rp.PopGenericObject();
    const auto buffer = rp.PopStaticBuffer();

    LOG_DEBUG(Service_APT, "called size={}", parameter_size);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(apt->applet_manager->LeaveHomeMenu(object, buffer));
}

void Module::APTInterface::LoadSysMenuArg(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x36, 1, 0); // 0x00360040
    const auto size = std::min(std::size_t{rp.Pop<u32>()}, SysMenuArgSize);

    LOG_DEBUG(Service_APT, "called");

    // This service function does not clear the buffer.
    std::vector<u8> buffer(size);
    std::copy_n(apt->sys_menu_arg_buffer.cbegin(), size, buffer.begin());

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 2);
    rb.Push(RESULT_SUCCESS);
    rb.PushStaticBuffer(std::move(buffer), 0);
}

void Module::APTInterface::StoreSysMenuArg(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x37, 1, 2); // 0x00370042
    const auto size = std::min(std::size_t{rp.Pop<u32>()}, SysMenuArgSize);
    const auto& buffer = rp.PopStaticBuffer();

    LOG_DEBUG(Service_APT, "called");
    ASSERT_MSG(buffer.size() >= size, "Buffer too small to hold requested data");

    std::copy_n(buffer.cbegin(), size, apt->sys_menu_arg_buffer.begin());

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(RESULT_SUCCESS);
}

void Module::APTInterface::SendCaptureBufferInfo(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x40, 1, 2); // 0x00400042
    const auto size = rp.Pop<u32>();
    const auto buffer = rp.PopStaticBuffer();

    LOG_DEBUG(Service_APT, "called");

    apt->applet_manager->SendCaptureBufferInfo(buffer);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(RESULT_SUCCESS);
}

void Module::APTInterface::ReceiveCaptureBufferInfo(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x41, 1, 0); // 0x00410040
    const auto size = rp.Pop<u32>();

    LOG_DEBUG(Service_APT, "called");

    auto screen_capture_buffer = apt->applet_manager->ReceiveCaptureBufferInfo();
    auto real_size = std::min(static_cast<u32>(screen_capture_buffer.size()), size);
    screen_capture_buffer.resize(size);

    IPC::RequestBuilder rb = rp.MakeBuilder(2, 2);
    rb.Push(RESULT_SUCCESS);
    rb.Push(real_size);
    rb.PushStaticBuffer(std::move(screen_capture_buffer), 0);
}

void Module::APTInterface::GetCaptureInfo(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x4A, 1, 0); // 0x004A0040
    const auto size = rp.Pop<u32>();

    LOG_DEBUG(Service_APT, "called");

    auto screen_capture_buffer = apt->applet_manager->GetCaptureInfo();
    auto real_size = std::min(static_cast<u32>(screen_capture_buffer.size()), size);
    screen_capture_buffer.resize(size);

    IPC::RequestBuilder rb = rp.MakeBuilder(2, 2);
    rb.Push(RESULT_SUCCESS);
    rb.Push(real_size);
    rb.PushStaticBuffer(std::move(screen_capture_buffer), 0);
}

void Module::APTInterface::SetScreenCapPostPermission(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x55, 1, 0); // 0x00550040

    LOG_DEBUG(Service_APT, "called, screen_capture_post_permission={}",
              apt->screen_capture_post_permission);

    apt->screen_capture_post_permission = static_cast<ScreencapPostPermission>(rp.Pop<u32>() & 0xF);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
    rb.Push(RESULT_SUCCESS); // No error
}

void Module::APTInterface::GetScreenCapPostPermission(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x56, 0, 0); // 0x00560000

    LOG_DEBUG(Service_APT, "(STUBBED) called, screen_capture_post_permission={}",
              apt->screen_capture_post_permission);

    IPC::RequestBuilder rb = rp.MakeBuilder(2, 0);
    rb.Push(RESULT_SUCCESS); // No error
    rb.Push(static_cast<u32>(apt->screen_capture_post_permission));
}

void Module::APTInterface::GetAppletInfo(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x6, 1, 0); // 0x60040
    const auto app_id = rp.PopEnum<AppletId>();

    LOG_DEBUG(Service_APT, "called, app_id={:08X}", app_id);

    auto info = apt->applet_manager->GetAppletInfo(app_id);
    if (info.Failed()) {
        IPC::RequestBuilder rb = rp.MakeBuilder(1, 0);
        rb.Push(info.Code());
    } else {
        IPC::RequestBuilder rb = rp.MakeBuilder(7, 0);
        rb.Push(RESULT_SUCCESS);
        rb.Push(info->title_id);
        rb.Push(static_cast<u8>(info->media_type));
        rb.Push(info->registered);
        rb.Push(info->loaded);
        rb.Push(info->attributes);
    }
}

void Module::APTInterface::GetStartupArgument(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x51, 2, 0); // 0x00510080
    const auto parameter_size = rp.Pop<u32>();
    const auto startup_argument_type = static_cast<StartupArgumentType>(rp.Pop<u8>());

    LOG_INFO(Service_APT, "called, startup_argument_type={}, parameter_size={:#010X}",
             startup_argument_type, parameter_size);

    std::vector<u8> param;
    bool exists = false;

    if (auto arg = apt->applet_manager->ReceiveDeliverArg()) {
        param = std::move(arg->param);

        // TODO: This is a complete guess based on observations. It is unknown how the OtherMedia
        // type is handled and how it interacts with the OtherApp type, and it is unknown if
        // this (checking the jump parameters) is indeed the way the 3DS checks the types.
        const auto& jump_parameters = apt->applet_manager->GetApplicationJumpParameters();
        switch (startup_argument_type) {
        case StartupArgumentType::OtherApp:
            exists = jump_parameters.current_title_id != jump_parameters.next_title_id &&
                     jump_parameters.current_media_type == jump_parameters.next_media_type;
            break;
        case StartupArgumentType::Restart:
            exists = jump_parameters.current_title_id == jump_parameters.next_title_id;
            break;
        case StartupArgumentType::OtherMedia:
            exists = jump_parameters.current_media_type != jump_parameters.next_media_type;
            break;
        }
    }

    constexpr u32 max_parameter_size{0x1000};
    param.resize(std::min(parameter_size, max_parameter_size));

    IPC::RequestBuilder rb = rp.MakeBuilder(2, 2);
    rb.Push(RESULT_SUCCESS);
    rb.Push(exists);
    rb.PushStaticBuffer(std::move(param), 0);
}

void Module::APTInterface::Wrap(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x46, 4, 4);
    const auto output_size = rp.Pop<u32>();
    const auto input_size = rp.Pop<u32>();
    const auto nonce_offset = rp.Pop<u32>();
    auto nonce_size = rp.Pop<u32>();
    auto& input = rp.PopMappedBuffer();
    auto& output = rp.PopMappedBuffer();

    LOG_DEBUG(Service_APT, "called, output_size={}, input_size={}, nonce_offset={}, nonce_size={}",
              output_size, input_size, nonce_offset, nonce_size);

    ASSERT(input.GetSize() == input_size);
    ASSERT(output.GetSize() == output_size);

    // Note: real 3DS still returns SUCCESS when the sizes don't match. It seems that it doesn't
    // check the buffer size and writes data with potential overflow.
    ASSERT_MSG(output_size == input_size + HW::AES::CCM_MAC_SIZE,
               "input_size ({}) doesn't match to output_size ({})", input_size, output_size);

    // Note: This weird nonce size modification is verified against real 3DS
    nonce_size = std::min<u32>(nonce_size & ~3, HW::AES::CCM_NONCE_SIZE);

    // Reads nonce and concatenates the rest of the input as plaintext
    HW::AES::CCMNonce nonce{};
    input.Read(nonce.data(), nonce_offset, nonce_size);
    u32 pdata_size = input_size - nonce_size;
    std::vector<u8> pdata(pdata_size);
    input.Read(pdata.data(), 0, nonce_offset);
    input.Read(pdata.data() + nonce_offset, nonce_offset + nonce_size, pdata_size - nonce_offset);

    // Encrypts the plaintext using AES-CCM
    auto cipher = HW::AES::EncryptSignCCM(pdata, nonce, HW::AES::KeySlotID::APTWrap);

    // Puts the nonce to the beginning of the output, with ciphertext followed
    output.Write(nonce.data(), 0, nonce_size);
    output.Write(cipher.data(), nonce_size, cipher.size());

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 4);
    rb.Push(RESULT_SUCCESS);
    // Unmap buffer
    rb.PushMappedBuffer(input);
    rb.PushMappedBuffer(output);
}

void Module::APTInterface::Unwrap(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x47, 4, 4);
    const auto output_size = rp.Pop<u32>();
    const auto input_size = rp.Pop<u32>();
    const auto nonce_offset = rp.Pop<u32>();
    auto nonce_size = rp.Pop<u32>();
    auto& input = rp.PopMappedBuffer();
    auto& output = rp.PopMappedBuffer();

    LOG_DEBUG(Service_APT, "called, output_size={}, input_size={}, nonce_offset={}, nonce_size={}",
              output_size, input_size, nonce_offset, nonce_size);

    ASSERT(input.GetSize() == input_size);
    ASSERT(output.GetSize() == output_size);

    // Note: real 3DS still returns SUCCESS when the sizes don't match. It seems that it doesn't
    // check the buffer size and writes data with potential overflow.
    ASSERT_MSG(output_size == input_size - HW::AES::CCM_MAC_SIZE,
               "input_size ({}) doesn't match to output_size ({})", input_size, output_size);

    // Note: This weird nonce size modification is verified against real 3DS
    nonce_size = std::min<u32>(nonce_size & ~3, HW::AES::CCM_NONCE_SIZE);

    // Reads nonce and cipher text
    HW::AES::CCMNonce nonce{};
    input.Read(nonce.data(), 0, nonce_size);
    u32 cipher_size = input_size - nonce_size;
    std::vector<u8> cipher(cipher_size);
    input.Read(cipher.data(), nonce_size, cipher_size);

    // Decrypts the ciphertext using AES-CCM
    auto pdata = HW::AES::DecryptVerifyCCM(cipher, nonce, HW::AES::KeySlotID::APTWrap);

    IPC::RequestBuilder rb = rp.MakeBuilder(1, 4);
    if (!pdata.empty()) {
        // Splits the plaintext and put the nonce in between
        output.Write(pdata.data(), 0, nonce_offset);
        output.Write(nonce.data(), nonce_offset, nonce_size);
        output.Write(pdata.data() + nonce_offset, nonce_offset + nonce_size,
                     pdata.size() - nonce_offset);
        rb.Push(RESULT_SUCCESS);
    } else {
        LOG_ERROR(Service_APT, "Failed to decrypt data");
        rb.Push(ResultCode(static_cast<ErrorDescription>(1), ErrorModule::PS,
                           ErrorSummary::WrongArgument, ErrorLevel::Status));
    }

    // Unmap buffer
    rb.PushMappedBuffer(input);
    rb.PushMappedBuffer(output);
}

void Module::APTInterface::CheckNew3DSApp(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x101, 0, 0); // 0x01010000

    LOG_WARNING(Service_APT, "(STUBBED) called");

    IPC::RequestBuilder rb = rp.MakeBuilder(2, 0);
    if (apt->unknown_ns_state_field) {
        rb.Push(RESULT_SUCCESS);
        rb.Push<u32>(0);
    } else {
        PTM::CheckNew3DS(rb);
    }
}

void Module::APTInterface::CheckNew3DS(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x102, 0, 0); // 0x01020000
    IPC::RequestBuilder rb = rp.MakeBuilder(2, 0);

    LOG_WARNING(Service_APT, "(STUBBED) called");

    PTM::CheckNew3DS(rb);
}

void Module::APTInterface::Unknown0x0103(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x103, 0, 0); // 0x01030000
    IPC::RequestBuilder rb = rp.MakeBuilder(2, 0);

    LOG_WARNING(Service_APT, "(STUBBED) called");

    rb.Push(RESULT_SUCCESS);
    rb.Push<u8>(Settings::values.is_new_3ds ? 2 : 1);
}

void Module::APTInterface::IsTitleAllowed(Kernel::HLERequestContext& ctx) {
    IPC::RequestParser rp(ctx, 0x105, 4, 0); // 0x01050100
    const auto program_id = rp.Pop<u64>();
    const auto media_type = static_cast<FS::MediaType>(rp.Pop<u8>());
    rp.Skip(1, false); // Padding

    LOG_DEBUG(Service_APT, "called, title_id={:016X} media_type={}", program_id, media_type);

    // We allow all titles to be launched, so this function is a no-op
    IPC::RequestBuilder rb = rp.MakeBuilder(2, 0);
    rb.Push(RESULT_SUCCESS);
    rb.Push(true);
}

Module::APTInterface::APTInterface(std::shared_ptr<Module> apt, const char* name, u32 max_session)
    : ServiceFramework(name, max_session), apt(std::move(apt)) {}

Module::APTInterface::~APTInterface() = default;

std::shared_ptr<Module> Module::APTInterface::GetModule() const {
    return apt;
}

Module::Module(Core::System& system) : system(system) {
    applet_manager = std::make_shared<AppletManager>(system);

    using Kernel::MemoryPermission;
    shared_font_mem = system.Kernel()
                          .CreateSharedMemory(nullptr, 0x332000, // 3272 KB
                                              MemoryPermission::ReadWrite, MemoryPermission::Read,
                                              0, Kernel::MemoryRegion::SYSTEM, "APT:SharedFont")
                          .Unwrap();
}

Module::~Module() {}

std::shared_ptr<AppletManager> Module::GetAppletManager() const {
    return applet_manager;
}

std::shared_ptr<Module> GetModule(Core::System& system) {
    auto apt = system.ServiceManager().GetService<Service::APT::Module::APTInterface>("APT:A");
    if (!apt)
        return nullptr;
    return apt->GetModule();
}

void InstallInterfaces(Core::System& system) {
    auto& service_manager = system.ServiceManager();
    auto apt = std::make_shared<Module>(system);
    std::make_shared<APT_U>(apt)->InstallAsService(service_manager);
    std::make_shared<APT_S>(apt)->InstallAsService(service_manager);
    std::make_shared<APT_A>(apt)->InstallAsService(service_manager);
    std::make_shared<Service::NS::NS_S>(apt)->InstallAsService(service_manager);
}

} // namespace Service::APT
