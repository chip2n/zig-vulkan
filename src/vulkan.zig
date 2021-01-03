usingnamespace @import("c.zig");
usingnamespace @import("utils.zig");

pub fn allocateCommandBuffers(
    device: VkDevice,
    info: *const VkCommandBufferAllocateInfo,
    buffers: [*c]VkCommandBuffer,
) !void {
    try checkSuccess(
        vkAllocateCommandBuffers(device, info, buffers),
        error.VulkanCommanbBufferAllocationFailure,
    );
}

pub fn queueSubmit(queue: VkQueue, submit_count: u32, submit_info: *const VkSubmitInfo, fence: ?VkFence) !void {
    try checkSuccess(
        vkQueueSubmit(queue, submit_count, submit_info, fence orelse null),
        error.VulkanQueueSubmitFailure,
    );
}

pub fn queueWaitIdle(queue: VkQueue) !void {
    try checkSuccess(vkQueueWaitIdle(queue), error.VulkanQueueWaitIdleFailure);
}

pub fn beginCommandBuffer(buffer: VkCommandBuffer, info: *const VkCommandBufferBeginInfo) !void {
    try checkSuccess(vkBeginCommandBuffer(buffer, info), error.VulkanBeginCommandBufferFailure);
}

pub fn endCommandBuffer(buffer: VkCommandBuffer) !void {
    try checkSuccess(vkEndCommandBuffer(buffer), error.VulkanCommandBufferEndFailure);
}
