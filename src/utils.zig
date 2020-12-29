const std = @import("std");

usingnamespace @import("c.zig");

pub fn checkSuccess(result: VkResult, comptime E: anytype) @TypeOf(E)!void {
    switch (result) {
        VkResult.VK_SUCCESS => {},
        else => return E,
    }
}

pub const CStrHashMap = std.HashMap(
    [*:0]const u8,
    void,
    hashCStr,
    eqlCStr,
    std.hash_map.DefaultMaxLoadPercentage,
);

fn hashCStr(a: [*:0]const u8) u64 {
    // FNV 32-bit hash
    var h: u32 = 2166136261;
    var i: usize = 0;
    while (a[i] != 0) : (i += 1) {
        h ^= a[i];
        h *%= 16777619;
    }
    return h;
}

fn eqlCStr(a: [*:0]const u8, b: [*:0]const u8) bool {
    return std.cstr.cmp(a, b) == 0;
}
