usingnamespace @import("c.zig");
usingnamespace @import("utils.zig");

pub fn allocateCommandBuffers(
    device: VkDevice,
    info: VkCommandBufferAllocateInfo,
    buffers: *VkCommandBuffer,
) !void {
    try checkSuccess(
        vkAllocateCommandBuffers(device, &info, buffers),
        error.VulkanCommanbBufferAllocationFailure,
    );
}
