const std = @import("std");
const Allocator = std.mem.Allocator;

usingnamespace @import("c.zig");
usingnamespace @import("utils.zig");

pub const QueueFamilyIndices = struct {
    graphics_family: ?u32,
    present_family: ?u32,

    pub fn isComplete(self: QueueFamilyIndices) bool {
        return self.graphics_family != null and self.present_family != null;
    }
};

pub fn findQueueFamilies(allocator: *Allocator, device: VkPhysicalDevice, surface: VkSurfaceKHR) !QueueFamilyIndices {
    var indices = QueueFamilyIndices{ .graphics_family = null, .present_family = null };

    var queue_family_count: u32 = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, null);

    const queue_families = try allocator.alloc(VkQueueFamilyProperties, queue_family_count);
    defer allocator.free(queue_families);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.ptr);

    // OPTIMIZE: use queue that supports all features if one is available
    var i: u32 = 0;
    for (queue_families) |family| {
        if (family.queueFlags & @intCast(u32, VK_QUEUE_GRAPHICS_BIT) != 0) {
            indices.graphics_family = @intCast(u32, i);
        }

        var present_support: VkBool32 = VK_FALSE;
        try checkSuccess(
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support),
            error.VulkanPresentSupportCheckFailed,
        );
        if (present_support == VK_TRUE) {
            indices.present_family = @intCast(u32, i);
        }

        if (indices.isComplete()) {
            break;
        }
        i += 1;
    }

    return indices;
}
