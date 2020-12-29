usingnamespace @import("c.zig");

pub fn checkSuccess(result: VkResult, comptime E: anytype) @TypeOf(E)!void {
    switch (result) {
        VkResult.VK_SUCCESS => {},
        else => return E,
    }
}
