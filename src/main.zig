const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const log = std.log;
const dbg = @import("debug.zig");

usingnamespace @import("c.zig");
usingnamespace @import("queue_family.zig");
usingnamespace @import("swap_chain.zig");
usingnamespace @import("utils.zig");

pub const log_level: std.log.Level = .warn;

const WIDTH = 800;
const HEIGHT = 600;

const enable_validation_layers = std.debug.runtime_safety;

const device_extensions = [_][*:0]const u8{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

fn System() type {
    if (builtin.mode == builtin.Mode.Debug) {
        return DebugSystem();
    } else {
        return ReleaseSystem();
    }
}

fn DebugSystem() type {
    return struct {
        const Self = @This();

        gpa: std.heap.GeneralPurposeAllocator(.{}),

        pub fn init() Self {
            return .{
                .gpa = std.heap.GeneralPurposeAllocator(.{}){},
            };
        }

        pub fn allocator(self: *Self) *Allocator {
            return &self.gpa.allocator;
        }

        pub fn deinit(self: *Self) void {
            _ = self.gpa.detectLeaks();
        }
    };
}

pub fn ReleaseSystem() type {
    return struct {
        const Self = @This();

        main_allocator: *Allocator,

        pub fn init() Self {
            return .{ .main_allocator = std.heap.c_allocator };
        }

        pub fn allocator(self: *Self) *Allocator {
            return self.main_allocator;
        }

        pub fn deinit(self: *Self) void {}
    };
}

pub fn main() !void {
    var system = System().init();
    defer system.deinit();
    const allocator = system.allocator();

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
    allocator: *Allocator,
    instance: VkInstance,
    physical_device: VkPhysicalDevice,
    logical_device: VkDevice,
    graphics_queue: VkQueue,
    present_queue: VkQueue,
    surface: VkSurfaceKHR,
    swap_chain: SwapChain,
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
        if (!indices.isComplete()) {
            return error.VulkanSuitableQueuFamiliesNotFound;
        }

        const logical_device = try createLogicalDevice(allocator, physical_device, indices);

        var graphics_queue: VkQueue = undefined;
        vkGetDeviceQueue(
            logical_device,
            indices.graphics_family.?,
            0,
            &graphics_queue,
        );

        var present_queue: VkQueue = undefined;
        vkGetDeviceQueue(
            logical_device,
            indices.present_family.?,
            0,
            &present_queue,
        );

        const swap_chain = try SwapChain.init(allocator, physical_device, logical_device, window, surface, indices);

        return Vulkan{
            .allocator = allocator,
            .instance = instance,
            .physical_device = physical_device,
            .logical_device = logical_device,
            .graphics_queue = graphics_queue,
            .present_queue = present_queue,
            .surface = surface,
            .swap_chain = swap_chain,
            .debug_messenger = debug_messenger,
        };
    }

    pub fn deinit(self: *const Vulkan) void {
        self.swap_chain.deinit(self.logical_device);
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
    const extensions_supported = try checkDeviceExtensionSupport(allocator, device);

    var swap_chain_adequate = false;
    if (extensions_supported) {
        const swap_chain_support = try querySwapChainSupport(allocator, device, surface);
        defer swap_chain_support.deinit();
        swap_chain_adequate = swap_chain_support.formats.len != 0 and swap_chain_support.present_modes.len != 0;
    }

    return indices.isComplete() and extensions_supported and swap_chain_adequate;
}

fn checkDeviceExtensionSupport(allocator: *Allocator, device: VkPhysicalDevice) !bool {
    var extension_count: u32 = 0;
    try checkSuccess(
        vkEnumerateDeviceExtensionProperties(device, null, &extension_count, null),
        error.VulkanExtensionPropsEnumerationFailed,
    );

    var available_extensions = try allocator.alloc(VkExtensionProperties, extension_count);
    defer allocator.free(available_extensions);
    try checkSuccess(
        vkEnumerateDeviceExtensionProperties(device, null, &extension_count, available_extensions.ptr),
        error.VulkanExtensionPropsEnumerationFailed,
    );

    var required_extensions = CStrHashMap.init(allocator);
    defer required_extensions.deinit();
    for (device_extensions) |extension| {
        _ = try required_extensions.put(extension, {});
    }

    for (available_extensions) |extension| {
        _ = required_extensions.remove(@ptrCast([*:0]const u8, &extension.extensionName));
    }

    return required_extensions.count() == 0;
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
        .ppEnabledExtensionNames = &device_extensions,
        .enabledExtensionCount = @intCast(u32, device_extensions.len),
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
