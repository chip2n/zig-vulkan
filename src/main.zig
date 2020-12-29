const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const log = std.log;

const dbg = @import("debug.zig");

usingnamespace @import("c.zig");
usingnamespace @import("utils.zig");

pub const log_level: std.log.Level = .warn;

const WIDTH = 800;
const HEIGHT = 600;

const enable_validation_layers = std.debug.runtime_safety;

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    const glfw = try GLFW.init();
    defer glfw.deinit();

    var vulkan = try Vulkan.init(allocator, glfw.window);
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
    physical_device: VkPhysicalDevice,
    logical_device: VkDevice,
    graphics_queue: VkQueue,
    present_queue: VkQueue,
    surface: VkSurfaceKHR,
    debug_messenger: ?VkDebugUtilsMessengerEXT,

    pub fn init(allocator: *Allocator, window: *GLFWwindow) !Vulkan {
        if (enable_validation_layers) {
            if (!try dbg.checkValidationLayerSupport(allocator)) {
                return error.ValidationLayerRequestedButNotAvailable;
            }
        }

        const app_info = VkApplicationInfo{
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

        var create_info = VkInstanceCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = @intCast(u32, extensions.len),
            .ppEnabledExtensionNames = extensions.ptr,
        };

        // placed outside scope to ensure it's not destroyed before the call to vkCreateInstance
        var debug_create_info: VkDebugUtilsMessengerCreateInfoEXT = undefined;
        if (enable_validation_layers) {
            debug_create_info = dbg.createDebugMessengerCreateInfo();
            dbg.fillDebugMessengerInInstanceCreateInfo(&create_info, &debug_create_info);
        }

        var instance: VkInstance = undefined;
        const result = vkCreateInstance(&create_info, null, &instance);
        if (result != VkResult.VK_SUCCESS) {
            return error.VulkanInitializationFailed;
        }

        var debug_messenger: VkDebugUtilsMessengerEXT = null;
        if (enable_validation_layers) {
            debug_messenger = try dbg.initDebugMessenger(instance);
        }

        const surface = try createSurface(instance, window);

        const physical_device = try pickPhysicalDevice(allocator, instance, surface);
        const indices = try findQueueFamilies(allocator, physical_device, surface);

        const logical_device = try createLogicalDevice(allocator, physical_device, indices);

        var graphics_queue: VkQueue = undefined;
        vkGetDeviceQueue(
            logical_device,
            indices.graphics_family.?, // TODO
            0,
            &graphics_queue,
        );

        var present_queue: VkQueue = undefined;
        vkGetDeviceQueue(
            logical_device,
            indices.present_family.?, // TODO
            0,
            &present_queue,
        );

        return Vulkan{
            .instance = instance,
            .physical_device = physical_device,
            .logical_device = logical_device,
            .graphics_queue = graphics_queue,
            .present_queue = present_queue,
            .surface = surface,
            .debug_messenger = debug_messenger,
        };
    }

    pub fn deinit(self: *const Vulkan) void {
        vkDestroyDevice(self.logical_device, null);
        if (self.debug_messenger) |messenger| {
            dbg.deinitDebugMessenger(self.instance, messenger);
        }
        vkDestroySurfaceKHR(self.instance, self.surface, null);
        vkDestroyInstance(self.instance, null);
    }
};

/// caller must free returned memory
fn getRequiredExtensions(allocator: *Allocator) ![][*:0]const u8 {
    var glfw_extension_count: u32 = 0;
    const glfw_extensions = @ptrCast([*]const [*:0]const u8, glfwGetRequiredInstanceExtensions(&glfw_extension_count));

    var extensions = ArrayList([*:0]const u8).init(allocator);
    errdefer extensions.deinit();

    try extensions.appendSlice(glfw_extensions[0..glfw_extension_count]);

    if (enable_validation_layers) {
        try extensions.append(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions.toOwnedSlice();
}

fn pickPhysicalDevice(allocator: *Allocator, instance: VkInstance, surface: VkSurfaceKHR) !VkPhysicalDevice {
    var device_count: u32 = 0;
    try checkSuccess(
        vkEnumeratePhysicalDevices(instance, &device_count, null),
        error.VulkanPhysicalDeviceEnumerationFailed,
    );

    if (device_count == 0) {
        return error.VulkanFailedToFindSupportedGPU;
    }

    const devices = try allocator.alloc(VkPhysicalDevice, device_count);
    defer allocator.free(devices);
    try checkSuccess(
        vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr),
        error.VulkanPhysicalDeviceEnumerationFailed,
    );

    const physical_device = for (devices) |device| {
        if (try isDeviceSuitable(allocator, device, surface)) {
            break device;
        }
    } else return error.VulkanFailedToFindSuitableGPU;

    return physical_device;
}

fn isDeviceSuitable(allocator: *Allocator, device: VkPhysicalDevice, surface: VkSurfaceKHR) !bool {
    const indices = try findQueueFamilies(allocator, device, surface);
    return indices.isComplete();
}

const QueueFamilyIndices = struct {
    graphics_family: ?u32,
    present_family: ?u32,

    pub fn isComplete(self: QueueFamilyIndices) bool {
        return self.graphics_family != null and self.present_family != null;
    }
};

fn findQueueFamilies(allocator: *Allocator, device: VkPhysicalDevice, surface: VkSurfaceKHR) !QueueFamilyIndices {
    var indices = QueueFamilyIndices{ .graphics_family = null, .present_family = null };

    var queue_family_count: u32 = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, null);

    const queue_families = try allocator.alloc(VkQueueFamilyProperties, queue_family_count);
    defer allocator.free(queue_families);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.ptr);

    // TODO(optimize): there might be a queue that supports all features, which would
    // be better for performance
    var i: u32 = 0;
    for (queue_families) |family| {
        if (family.queueFlags & @intCast(u32, VK_QUEUE_GRAPHICS_BIT) == 0) {
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

fn createLogicalDevice(allocator: *Allocator, physical_device: VkPhysicalDevice, indices: QueueFamilyIndices) !VkDevice {
    const all_queue_families = [_]u32{ indices.graphics_family.?, indices.present_family.? };
    const unique_queue_families = if (indices.graphics_family.? == indices.present_family.?)
        all_queue_families[0..1]
    else
        all_queue_families[0..2];

    var queue_create_infos = ArrayList(VkDeviceQueueCreateInfo).init(allocator);
    defer queue_create_infos.deinit();

    var queue_priority: f32 = 1.0;
    for (unique_queue_families) |queue_family| {
        const queue_create_info = VkDeviceQueueCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueFamilyIndex = queue_family,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };
        try queue_create_infos.append(queue_create_info);
    }

    const device_features = VkPhysicalDeviceFeatures{
        .robustBufferAccess = 0,
        .fullDrawIndexUint32 = 0,
        .imageCubeArray = 0,
        .independentBlend = 0,
        .geometryShader = 0,
        .tessellationShader = 0,
        .sampleRateShading = 0,
        .dualSrcBlend = 0,
        .logicOp = 0,
        .multiDrawIndirect = 0,
        .drawIndirectFirstInstance = 0,
        .depthClamp = 0,
        .depthBiasClamp = 0,
        .fillModeNonSolid = 0,
        .depthBounds = 0,
        .wideLines = 0,
        .largePoints = 0,
        .alphaToOne = 0,
        .multiViewport = 0,
        .samplerAnisotropy = 0,
        .textureCompressionETC2 = 0,
        .textureCompressionASTC_LDR = 0,
        .textureCompressionBC = 0,
        .occlusionQueryPrecise = 0,
        .pipelineStatisticsQuery = 0,
        .vertexPipelineStoresAndAtomics = 0,
        .fragmentStoresAndAtomics = 0,
        .shaderTessellationAndGeometryPointSize = 0,
        .shaderImageGatherExtended = 0,
        .shaderStorageImageExtendedFormats = 0,
        .shaderStorageImageMultisample = 0,
        .shaderStorageImageReadWithoutFormat = 0,
        .shaderStorageImageWriteWithoutFormat = 0,
        .shaderUniformBufferArrayDynamicIndexing = 0,
        .shaderSampledImageArrayDynamicIndexing = 0,
        .shaderStorageBufferArrayDynamicIndexing = 0,
        .shaderStorageImageArrayDynamicIndexing = 0,
        .shaderClipDistance = 0,
        .shaderCullDistance = 0,
        .shaderFloat64 = 0,
        .shaderInt64 = 0,
        .shaderInt16 = 0,
        .shaderResourceResidency = 0,
        .shaderResourceMinLod = 0,
        .sparseBinding = 0,
        .sparseResidencyBuffer = 0,
        .sparseResidencyImage2D = 0,
        .sparseResidencyImage3D = 0,
        .sparseResidency2Samples = 0,
        .sparseResidency4Samples = 0,
        .sparseResidency8Samples = 0,
        .sparseResidency16Samples = 0,
        .sparseResidencyAliased = 0,
        .variableMultisampleRate = 0,
        .inheritedQueries = 0,
    };
    var create_info = VkDeviceCreateInfo{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .pQueueCreateInfos = queue_create_infos.items.ptr,
        .queueCreateInfoCount = @intCast(u32, queue_create_infos.items.len),
        .pEnabledFeatures = &device_features,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
    };

    if (enable_validation_layers) {
        dbg.fillDebugMessengerInDeviceCreateInfo(&create_info);
    }

    var device: VkDevice = undefined;

    try checkSuccess(
        vkCreateDevice(physical_device, &create_info, null, &device),
        error.VulkanLogicalDeviceCreationFailed,
    );

    return device;
}

fn createSurface(instance: VkInstance, window: *GLFWwindow) !VkSurfaceKHR {
    var surface: VkSurfaceKHR = undefined;
    try checkSuccess(
        glfwCreateWindowSurface(instance, window, null, &surface),
        error.VulkanWindowSurfaceCreationFailed,
    );
    return surface;
}
