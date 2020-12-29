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

const enableValidationLayers = std.debug.runtime_safety;

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
    physicalDevice: VkPhysicalDevice,
    logicalDevice: VkDevice,
    graphicsQueue: VkQueue,
    debugMessenger: ?VkDebugUtilsMessengerEXT,

    pub fn init(allocator: *Allocator) !Vulkan {
        if (enableValidationLayers) {
            if (!try dbg.checkValidationLayerSupport(allocator)) {
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

        var createInfo = VkInstanceCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = @intCast(u32, extensions.len),
            .ppEnabledExtensionNames = extensions.ptr,
        };

        // placed outside scope to ensure it's not destroyed before the call to vkCreateInstance
        var debugCreateInfo: VkDebugUtilsMessengerCreateInfoEXT = undefined;
        if (enableValidationLayers) {
            debugCreateInfo = dbg.createDebugMessengerCreateInfo();
            dbg.fillDebugMessengerInInstanceCreateInfo(&createInfo, &debugCreateInfo);
        }

        var instance: VkInstance = undefined;
        const result = vkCreateInstance(&createInfo, null, &instance);
        if (result != VkResult.VK_SUCCESS) {
            return error.VulkanInitializationFailed;
        }

        var debugMessenger: VkDebugUtilsMessengerEXT = null;
        if (enableValidationLayers) {
            debugMessenger = try dbg.initDebugMessenger(instance);
        }

        const physicalDevice = try pickPhysicalDevice(allocator, instance);
        const logicalDevice = try createLogicalDevice(allocator, physicalDevice);

        const indices = try findQueueFamilies(allocator, physicalDevice); // TODO reuse

        var graphicsQueue: VkQueue = undefined;
        vkGetDeviceQueue(
            logicalDevice,
            indices.graphicsFamily.?, // TODO
            0,
            &graphicsQueue,
        );

        return Vulkan{
            .instance = instance,
            .physicalDevice = physicalDevice,
            .logicalDevice = logicalDevice,
            .graphicsQueue = graphicsQueue,
            .debugMessenger = debugMessenger,
        };
    }

    pub fn deinit(self: *const Vulkan) void {
        vkDestroyDevice(self.logicalDevice, null);
        if (self.debugMessenger) |messenger| {
            dbg.deinitDebugMessenger(self.instance, messenger);
        }
        vkDestroyInstance(self.instance, null);
    }
};

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

fn pickPhysicalDevice(allocator: *Allocator, instance: VkInstance) !VkPhysicalDevice {
    var deviceCount: u32 = 0;
    try checkSuccess(
        vkEnumeratePhysicalDevices(instance, &deviceCount, null),
        error.VulkanPhysicalDeviceEnumerationFailed,
    );

    if (deviceCount == 0) {
        return error.VulkanFailedToFindSupportedGPU;
    }

    const devices = try allocator.alloc(VkPhysicalDevice, deviceCount);
    defer allocator.free(devices);
    try checkSuccess(
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.ptr),
        error.VulkanPhysicalDeviceEnumerationFailed,
    );

    const physicalDevice = for (devices) |device| {
        if (try isDeviceSuitable(allocator, device)) {
            break device;
        }
    } else return error.VulkanFailedToFindSuitableGPU;

    return physicalDevice;
}

fn isDeviceSuitable(allocator: *Allocator, device: VkPhysicalDevice) !bool {
    const indices = try findQueueFamilies(allocator, device);
    return indices.isComplete();
}

const QueueFamilyIndices = struct {
    graphicsFamily: ?u32,

    pub fn isComplete(self: QueueFamilyIndices) bool {
        return self.graphicsFamily != null;
    }
};

fn findQueueFamilies(allocator: *Allocator, device: VkPhysicalDevice) !QueueFamilyIndices {
    var indices = QueueFamilyIndices{ .graphicsFamily = null };

    var queueFamilyCount: u32 = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, null);

    const queueFamilies = try allocator.alloc(VkQueueFamilyProperties, queueFamilyCount);
    defer allocator.free(queueFamilies);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.ptr);

    for (queueFamilies) |family, i| {
        if (family.queueFlags & @intCast(u32, VK_QUEUE_GRAPHICS_BIT) == 0) {
            indices.graphicsFamily = @intCast(u32, i);
        }

        if (indices.isComplete()) {
            break;
        }
    }

    return indices;
}

fn createLogicalDevice(allocator: *Allocator, physicalDevice: VkPhysicalDevice) !VkDevice {
    const indices = try findQueueFamilies(allocator, physicalDevice); // TODO reuse
    var queuePriorities: f32 = 1.0;
    const queueCreateInfo = VkDeviceQueueCreateInfo{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .queueFamilyIndex = indices.graphicsFamily.?, // TODO optional
        .queueCount = 1,
        .pQueuePriorities = &queuePriorities,
    };

    const deviceFeatures = VkPhysicalDeviceFeatures{
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
    var createInfo = VkDeviceCreateInfo{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .pQueueCreateInfos = &queueCreateInfo,
        .queueCreateInfoCount = 1,
        .pEnabledFeatures = &deviceFeatures,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
    };

    if (enableValidationLayers) {
        dbg.fillDebugMessengerInDeviceCreateInfo(&createInfo);
    }

    var device: VkDevice = undefined;

    try checkSuccess(
        vkCreateDevice(physicalDevice, &createInfo, null, &device),
        error.VulkanLogicalDeviceCreationFailed,
    );

    return device;
}
