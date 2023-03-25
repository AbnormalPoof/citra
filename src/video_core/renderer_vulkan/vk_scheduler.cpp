// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <mutex>
#include <utility>
#include "common/microprofile.h"
#include "common/settings.h"
#include "common/thread.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

MICROPROFILE_DEFINE(Vulkan_WaitForWorker, "Vulkan", "Wait for worker", MP_RGB(255, 192, 192));
MICROPROFILE_DEFINE(Vulkan_Submit, "Vulkan", "Submit Exectution", MP_RGB(255, 192, 255));

namespace Vulkan {

void Scheduler::CommandChunk::ExecuteAll(vk::CommandBuffer cmdbuf) {
    auto command = first;
    while (command != nullptr) {
        auto next = command->GetNext();
        command->Execute(cmdbuf);
        command->~Command();
        command = next;
    }
    submit = false;
    command_offset = 0;
    first = nullptr;
    last = nullptr;
}

Scheduler::Scheduler(const Instance& instance, RenderpassCache& renderpass_cache)
    : instance{instance}, renderpass_cache{renderpass_cache}, master_semaphore{instance},
      command_pool{instance, master_semaphore}, use_worker_thread{
                                                    !Settings::values.renderer_debug} {
    AllocateWorkerCommandBuffers();
    if (use_worker_thread) {
        AcquireNewChunk();
        worker_thread = std::jthread([this](std::stop_token token) { WorkerThread(token); });
    }
}

Scheduler::~Scheduler() = default;

void Scheduler::Flush(vk::Semaphore signal, vk::Semaphore wait) {
    // When flushing, we only send data to the worker thread; no waiting is necessary.
    SubmitExecution(signal, wait);
}

void Scheduler::Finish(vk::Semaphore signal, vk::Semaphore wait) {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitExecution(signal, wait);
    Wait(presubmit_tick);
}

void Scheduler::WaitWorker() {
    if (!use_worker_thread) {
        return;
    }

    MICROPROFILE_SCOPE(Vulkan_WaitForWorker);
    DispatchWork();

    // Ensure the queue is drained.
    {
        std::unique_lock ql{queue_mutex};
        event_cv.wait(ql, [this] { return work_queue.empty(); });
    }

    // Now wait for execution to finish.
    // This needs to be done in the same order as WorkerThread.
    std::scoped_lock el{execution_mutex};
}

void Scheduler::Wait(u64 tick) {
    if (tick >= master_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        Flush();
    }
    master_semaphore.Wait(tick);
}

void Scheduler::DispatchWork() {
    if (!use_worker_thread || chunk->Empty()) {
        return;
    }

    {
        std::scoped_lock ql{queue_mutex};
        work_queue.push(std::move(chunk));
    }

    event_cv.notify_all();
    AcquireNewChunk();
}

void Scheduler::WorkerThread(std::stop_token stop_token) {
    Common::SetCurrentThreadName("VulkanWorker");

    const auto TryPopQueue{[this](auto& work) -> bool {
        if (work_queue.empty()) {
            return false;
        }

        work = std::move(work_queue.front());
        work_queue.pop();
        event_cv.notify_all();
        return true;
    }};

    while (!stop_token.stop_requested()) {
        std::unique_ptr<CommandChunk> work;

        {
            std::unique_lock lk{queue_mutex};

            // Wait for work.
            Common::CondvarWait(event_cv, lk, stop_token, [&] { return TryPopQueue(work); });

            // If we've been asked to stop, we're done.
            if (stop_token.stop_requested()) {
                return;
            }

            // Exchange lock ownership so that we take the execution lock before
            // the queue lock goes out of scope. This allows us to force execution
            // to complete in the next step.
            std::exchange(lk, std::unique_lock{execution_mutex});

            // Perform the work, tracking whether the chunk was a submission
            // before executing.
            const bool has_submit = work->HasSubmit();
            work->ExecuteAll(current_cmdbuf);

            // If the chunk was a submission, reallocate the command buffer.
            if (has_submit) {
                AllocateWorkerCommandBuffers();
            }
        }

        {
            std::scoped_lock rl{reserve_mutex};

            // Recycle the chunk back to the reserve.
            chunk_reserve.emplace_back(std::move(work));
        }
    }
}

void Scheduler::AllocateWorkerCommandBuffers() {
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    current_cmdbuf = command_pool.Commit();
    current_cmdbuf.begin(begin_info);
}

void Scheduler::SubmitExecution(vk::Semaphore signal_semaphore, vk::Semaphore wait_semaphore) {
    const vk::Semaphore handle = master_semaphore.Handle();
    const u64 signal_value = master_semaphore.NextTick();
    state = StateFlags::AllDirty;

    renderpass_cache.EndRendering();
    Record(
        [signal_semaphore, wait_semaphore, handle, signal_value, this](vk::CommandBuffer cmdbuf) {
            MICROPROFILE_SCOPE(Vulkan_Submit);
            cmdbuf.end();

            const u32 num_signal_semaphores = signal_semaphore ? 2U : 1U;
            const std::array signal_values{signal_value, u64(0)};
            const std::array signal_semaphores{handle, signal_semaphore};

            const u32 num_wait_semaphores = wait_semaphore ? 2U : 1U;
            const std::array wait_values{signal_value - 1, u64(1)};
            const std::array wait_semaphores{handle, wait_semaphore};

            static constexpr std::array<vk::PipelineStageFlags, 2> wait_stage_masks = {
                vk::PipelineStageFlagBits::eAllCommands,
                vk::PipelineStageFlagBits::eColorAttachmentOutput,
            };

            const vk::TimelineSemaphoreSubmitInfoKHR timeline_si = {
                .waitSemaphoreValueCount = num_wait_semaphores,
                .pWaitSemaphoreValues = wait_values.data(),
                .signalSemaphoreValueCount = num_signal_semaphores,
                .pSignalSemaphoreValues = signal_values.data(),
            };

            const vk::SubmitInfo submit_info = {
                .pNext = &timeline_si,
                .waitSemaphoreCount = num_wait_semaphores,
                .pWaitSemaphores = wait_semaphores.data(),
                .pWaitDstStageMask = wait_stage_masks.data(),
                .commandBufferCount = 1u,
                .pCommandBuffers = &cmdbuf,
                .signalSemaphoreCount = num_signal_semaphores,
                .pSignalSemaphores = signal_semaphores.data(),
            };

            try {
                std::scoped_lock lock{queue_mutex};
                instance.GetGraphicsQueue().submit(submit_info);
            } catch (vk::DeviceLostError& err) {
                LOG_CRITICAL(Render_Vulkan, "Device lost during submit: {}", err.what());
                UNREACHABLE();
            }
        });

    if (!use_worker_thread) {
        AllocateWorkerCommandBuffers();
    } else {
        chunk->MarkSubmit();
        DispatchWork();
    }
}

void Scheduler::AcquireNewChunk() {
    std::scoped_lock lock{reserve_mutex};
    if (chunk_reserve.empty()) {
        chunk = std::make_unique<CommandChunk>();
        return;
    }

    chunk = std::move(chunk_reserve.back());
    chunk_reserve.pop_back();
}

} // namespace Vulkan
