const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const log = std.log;

usingnamespace @import("c.zig");

const WIDTH = 800;
const HEIGHT = 600;

const enableValidationLayers = std.debug.runtime_safety;
const validationLayers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};

fn debugCallback(
    messageSeverity: VkDebugUtilsMessageSeverityFlagBitsEXT,
    messageType: VkDebugUtilsMessageTypeFlagsEXT,
    pCallbackData: [*c]const VkDebugUtilsMessengerCallbackDataEXT,
    pUserData: ?*c_void,
) callconv(.C) u32 {
    const msg = @ptrCast([*:0]const u8, pCallbackData.*.pMessage);
    log.err("validation layer: {}", .{msg});

    return VK_FALSE;
}

fn CreateDebugUtilsMessengerEXT(
    instance: VkInstance,
    pCreateInfo: *const VkDebugUtilsMessengerCreateInfoEXT,
    pAllocator: ?*const VkAllocationCallbacks,
    pDebugMessenger: *VkDebugUtilsMessengerEXT,
) VkResult {
    const func = @ptrCast(PFN_vkCreateDebugUtilsMessengerEXT, vkGetInstanceProcAddr(
        instance,
        "vkCreateDebugUtilsMessengerEXT",
    )) orelse return VkResult.VK_ERROR_EXTENSION_NOT_PRESENT;

    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
}

fn DestroyDebugUtilsMessengerEXT(
    instance: VkInstance,
    debugMessenger: VkDebugUtilsMessengerEXT,
    pAllocator: ?*const VkAllocationCallbacks,
) void {
    const optional_func = @ptrCast(PFN_vkDestroyDebugUtilsMessengerEXT, vkGetInstanceProcAddr(
        instance,
        "vkDestroyDebugUtilsMessengerEXT",
    ));

    if (optional_func) |func| {
        return func(instance, debugMessenger, pAllocator);
    }
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    const glfw = try GLFW.init();
    defer glfw.deinit();

    var vulkan = try Vulkan.init(allocator);
    defer vulkan.deinit();

    while (glfwWindowShouldClose(glfw.window) == GLFW_FALSE) {
        glfwPollEvents();
    }
}

const GLFW = struct {
    window: *GLFWwindow,

    pub fn init() !GLFW {
        const init_result = glfwInit();
        if (init_result == GLFW_FALSE) {
            return error.GLFWInitializationFailed;
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        const window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan window", null, null);
        if (window == null) {
            return error.GLFWInitializationFailed;
        }

        return GLFW{ .window = window.? };
    }

    pub fn deinit(self: *const GLFW) void {
        glfwDestroyWindow(self.window);
        glfwTerminate();
    }
};

const Vulkan = struct {
    instance: VkInstance,
    debugMessenger: ?VkDebugUtilsMessengerEXT,

    pub fn init(allocator: *Allocator) !Vulkan {
        if (enableValidationLayers) {
            if (!try checkValidationLayerSupport(allocator)) {
                return error.ValidationLayerRequestedButNotAvailable;
            }
        }

        const appInfo = VkApplicationInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = null,
            .pApplicationName = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_0,
        };

        const extensions = try getRequiredExtensions(allocator);
        defer allocator.free(extensions);

        const createInfo = VkInstanceCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = if (enableValidationLayers) validationLayers.len else 0,
            .ppEnabledLayerNames = if (enableValidationLayers) &validationLayers else null,
            .enabledExtensionCount = @intCast(u32, extensions.len),
            .ppEnabledExtensionNames = extensions.ptr,
        };

        var instance: VkInstance = undefined;
        const result = vkCreateInstance(&createInfo, null, &instance);
        if (result != VkResult.VK_SUCCESS) {
            return error.VulkanInitializationFailed;
        }

        const debugMessenger = try initDebugMessenger(instance);

        return Vulkan{
            .instance = instance,
            .debugMessenger = debugMessenger,
        };
    }

    fn initDebugMessenger(instance: VkInstance) !VkDebugUtilsMessengerEXT {
        if (!enableValidationLayers) return;

        const createInfo = VkDebugUtilsMessengerCreateInfoEXT{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .pNext = null,
            .flags = 0,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback,
            .pUserData = null,
        };

        var debugMessenger: VkDebugUtilsMessengerEXT = undefined;
        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, null, &debugMessenger) != VkResult.VK_SUCCESS) {
            return error.VulkanDebugMessengerSetupFailed;
        }

        return debugMessenger;
    }

    pub fn deinit(self: *const Vulkan) void {
        if (self.debugMessenger) |messenger| {
            DestroyDebugUtilsMessengerEXT(self.instance, messenger, null);
        }
        vkDestroyInstance(self.instance, null);
    }
};

fn checkValidationLayerSupport(allocator: *Allocator) !bool {
    var layerCount: u32 = undefined;

    try checkSuccess(vkEnumerateInstanceLayerProperties(&layerCount, null));

    const availableLayers = try allocator.alloc(VkLayerProperties, layerCount);
    defer allocator.free(availableLayers);

    try checkSuccess(vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.ptr));

    for (validationLayers) |layerName| {
        var layerFound = false;

        for (availableLayers) |layerProperties| {
            if (std.cstr.cmp(layerName, @ptrCast([*:0]const u8, &layerProperties.layerName)) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}

/// caller must free returned memory
fn getRequiredExtensions(allocator: *Allocator) ![][*:0]const u8 {
    var glfwExtensionCount: u32 = 0;
    const glfwExtensions = @ptrCast([*]const [*:0]const u8, glfwGetRequiredInstanceExtensions(&glfwExtensionCount));

    var extensions = ArrayList([*:0]const u8).init(allocator);
    errdefer extensions.deinit();

    try extensions.appendSlice(glfwExtensions[0..glfwExtensionCount]);

    if (enableValidationLayers) {
        try extensions.append(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions.toOwnedSlice();
}

fn checkSuccess(result: VkResult) !void {
    switch (result) {
        VkResult.VK_SUCCESS => {},
        else => return error.Unexpected,
    }
}
