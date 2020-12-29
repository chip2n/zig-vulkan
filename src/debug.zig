const std = @import("std");
const Allocator = std.mem.Allocator;

const log = std.log.scoped(.vulkan_validation_layer);

usingnamespace @import("c.zig");
usingnamespace @import("utils.zig");

const validationLayers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};

pub fn initDebugMessenger(instance: VkInstance) !VkDebugUtilsMessengerEXT {
    const createInfo = createDebugMessengerCreateInfo();

    var debugMessenger: VkDebugUtilsMessengerEXT = undefined;
    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, null, &debugMessenger) != VkResult.VK_SUCCESS) {
        return error.VulkanDebugMessengerSetupFailed;
    }

    return debugMessenger;
}

pub fn deinitDebugMessenger(instance: VkInstance, debugMessenger: VkDebugUtilsMessengerEXT) void {
    DestroyDebugUtilsMessengerEXT(instance, debugMessenger, null);
}

pub fn checkValidationLayerSupport(allocator: *Allocator) !bool {
    var layerCount: u32 = undefined;

    try checkSuccess(
        vkEnumerateInstanceLayerProperties(&layerCount, null),
        error.VulkanLayerPropsEnumerationFailed,
    );

    const availableLayers = try allocator.alloc(VkLayerProperties, layerCount);
    defer allocator.free(availableLayers);

    try checkSuccess(
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.ptr),
        error.VulkanLayerPropsEnumerationFailed,
    );

    if (checkSuccess(
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.ptr),
        error.VulkanLayerPropsEnumerationFailed,
    )) {} else |err| switch (err) {
        error.VulkanLayerPropsEnumerationFailed => log.warn("yee", .{}),
    }

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

pub fn createDebugMessengerCreateInfo() VkDebugUtilsMessengerCreateInfoEXT {
    return VkDebugUtilsMessengerCreateInfoEXT{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = null,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
        .pUserData = null,
    };
}

pub fn fillDebugMessengerInInstanceCreateInfo(createInfo: *VkInstanceCreateInfo, debugCreateInfo: *VkDebugUtilsMessengerCreateInfoEXT) void {
    createInfo.enabledLayerCount = validationLayers.len;
    createInfo.ppEnabledLayerNames = &validationLayers;
    createInfo.pNext = debugCreateInfo;
}

pub fn fillDebugMessengerInDeviceCreateInfo(createInfo: *VkDeviceCreateInfo) void {
    createInfo.enabledLayerCount = validationLayers.len;
    createInfo.ppEnabledLayerNames = &validationLayers;
}

fn debugCallback(
    messageSeverity: VkDebugUtilsMessageSeverityFlagBitsEXT,
    messageType: VkDebugUtilsMessageTypeFlagsEXT,
    pCallbackData: [*c]const VkDebugUtilsMessengerCallbackDataEXT,
    pUserData: ?*c_void,
) callconv(.C) u32 {
    const msg = @ptrCast([*:0]const u8, pCallbackData.*.pMessage);

    const severity = @enumToInt(messageSeverity);
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        log.err("{}", .{msg});
    } else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        log.warn("{}", .{msg});
    } else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        log.info("{}", .{msg});
    } else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
        log.debug("{}", .{msg});
    }

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
