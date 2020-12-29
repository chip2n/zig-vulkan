const std = @import("std");
const Allocator = std.mem.Allocator;

const log = std.log.scoped(.vulkan_validation_layer);

usingnamespace @import("c.zig");
usingnamespace @import("utils.zig");

const validation_layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};

pub fn initDebugMessenger(instance: VkInstance) !VkDebugUtilsMessengerEXT {
    const create_info = createDebugMessengerCreateInfo();

    var debug_messenger: VkDebugUtilsMessengerEXT = undefined;
    if (CreateDebugUtilsMessengerEXT(instance, &create_info, null, &debug_messenger) != VkResult.VK_SUCCESS) {
        return error.VulkanDebugMessengerSetupFailed;
    }

    return debug_messenger;
}

pub fn deinitDebugMessenger(instance: VkInstance, debug_messenger: VkDebugUtilsMessengerEXT) void {
    DestroyDebugUtilsMessengerEXT(instance, debug_messenger, null);
}

pub fn checkValidationLayerSupport(allocator: *Allocator) !bool {
    var layer_count: u32 = undefined;

    try checkSuccess(
        vkEnumerateInstanceLayerProperties(&layer_count, null),
        error.VulkanLayerPropsEnumerationFailed,
    );

    const available_layers = try allocator.alloc(VkLayerProperties, layer_count);
    defer allocator.free(available_layers);

    try checkSuccess(
        vkEnumerateInstanceLayerProperties(&layer_count, available_layers.ptr),
        error.VulkanLayerPropsEnumerationFailed,
    );

    if (checkSuccess(
        vkEnumerateInstanceLayerProperties(&layer_count, available_layers.ptr),
        error.VulkanLayerPropsEnumerationFailed,
    )) {} else |err| switch (err) {
        error.VulkanLayerPropsEnumerationFailed => log.warn("yee", .{}),
    }

    for (validation_layers) |layer_name| {
        var layer_found = false;

        for (available_layers) |layer_properties| {
            if (std.cstr.cmp(layer_name, @ptrCast([*:0]const u8, &layer_properties.layerName)) == 0) {
                layer_found = true;
                break;
            }
        }

        if (!layer_found) {
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

pub fn fillDebugMessengerInInstanceCreateInfo(create_info: *VkInstanceCreateInfo, debug_create_info: *VkDebugUtilsMessengerCreateInfoEXT) void {
    create_info.enabledLayerCount = validation_layers.len;
    create_info.ppEnabledLayerNames = &validation_layers;
    create_info.pNext = debug_create_info;
}

pub fn fillDebugMessengerInDeviceCreateInfo(create_info: *VkDeviceCreateInfo) void {
    create_info.enabledLayerCount = validation_layers.len;
    create_info.ppEnabledLayerNames = &validation_layers;
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
