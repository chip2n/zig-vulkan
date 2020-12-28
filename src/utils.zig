usingnamespace @import("c.zig");

pub fn checkSuccess(result: VkResult) !void {
    switch (result) {
        VkResult.VK_SUCCESS => {},
        else => return error.Unexpected,
    }
}
