usingnamespace @import("c.zig");
usingnamespace @import("utils.zig");

const WIDTH = 800;
const HEIGHT = 600;

pub const ResizeCallback = struct {
    data: *c_void,
    cb: fn (*c_void) void,
};

pub const Window = struct {
    const Self = @This();

    window: *GLFWwindow,

    pub fn init() !Self {
        const init_result = glfwInit();
        if (init_result == GLFW_FALSE) {
            return error.GLFWInitializationFailed;
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        const window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan window", null, null);
        if (window == null) {
            return error.GLFWInitializationFailed;
        }

        return Self{ .window = window.? };
    }

    pub fn deinit(self: *const Self) void {
        glfwDestroyWindow(self.window);
        glfwTerminate();
    }

    pub fn registerResizeCallback(self: *Self, callback: *ResizeCallback) void {
        glfwSetWindowUserPointer(self.window, callback);
        _ = glfwSetFramebufferSizeCallback(self.window, framebufferResizeCallback);
    }

    pub fn getFramebufferSize(self: *const Self) Size {
        var width: c_int = 0;
        var height: c_int = 0;
        glfwGetFramebufferSize(self.window, &width, &height);
        return Size{
            .width = @intCast(u32, width),
            .height = @intCast(u32, height),
        };
    }

    pub fn createSurface(self: *const Self, instance: VkInstance) !VkSurfaceKHR {
        var surface: VkSurfaceKHR = undefined;
        try checkSuccess(
            glfwCreateWindowSurface(instance, self.window, null, &surface),
            error.VulkanWindowSurfaceCreationFailed,
        );
        return surface;
    }
};

fn framebufferResizeCallback(window: ?*GLFWwindow, width: c_int, height: c_int) callconv(.C) void {
    var callback = @ptrCast(*ResizeCallback, @alignCast(@alignOf(*ResizeCallback), glfwGetWindowUserPointer(window)));
    callback.cb(callback.data);
}
