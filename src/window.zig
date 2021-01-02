usingnamespace @import("c.zig");
usingnamespace @import("utils.zig");

const WIDTH = 800;
const HEIGHT = 600;

pub const ResizeCallback = struct {
    data: *c_void,
    cb: fn(*c_void) void,
};

pub const GLFW = struct {
    window: *GLFWwindow,

    pub fn init() !GLFW {
        const init_result = glfwInit();
        if (init_result == GLFW_FALSE) {
            return error.GLFWInitializationFailed;
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        const window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan window", null, null);
        if (window == null) {
            return error.GLFWInitializationFailed;
        }

        return GLFW{ .window = window.? };
    }

    pub fn registerResizeCallback(self: *@This(), callback: *ResizeCallback) void {
        glfwSetWindowUserPointer(self.window, callback);
        _ = glfwSetFramebufferSizeCallback(self.window, framebufferResizeCallback);
    }

    pub fn deinit(self: *const GLFW) void {
        glfwDestroyWindow(self.window);
        glfwTerminate();
    }
};

fn framebufferResizeCallback(window: ?*GLFWwindow, width: c_int, height: c_int) callconv(.C) void {
    var callback = @ptrCast(*ResizeCallback, @alignCast(@alignOf(*ResizeCallback), glfwGetWindowUserPointer(window)));
    callback.cb(callback.data);
}

pub fn getFramebufferSize(window: *GLFWwindow) Size {
    var width: c_int = 0;
    var height: c_int = 0;
    glfwGetFramebufferSize(window, &width, &height);
    return Size{
        .width = @intCast(u32, width),
        .height = @intCast(u32, height),
    };
}
