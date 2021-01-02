const std = @import("std");
const mem = std.mem;
const builtin = @import("builtin");
const Allocator = mem.Allocator;
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
const MAX_FRAMES_IN_FLIGHT = 2;
const MAX_UINT64 = @as(c_ulong, 18446744073709551615); // Couldn't use UINT64_MAX for some reason

const enable_validation_layers = std.debug.runtime_safety;

const device_extensions = [_][*:0]const u8{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

// TODO make non-global
var frame_buffer_resized = false;

pub fn main() !void {
    var system = System().init();
    defer system.deinit();

    const allocator = system.allocator();

    var context = try RenderContext.init(allocator);
    defer context.deinit();

    while (!context.shouldClose()) {
        try context.renderFrame();
    }
}

fn System() type {
    if (builtin.mode == builtin.Mode.Debug) {
        return DebugSystem();
    } else {
        return ReleaseSystem();
    }
}

/// uses the general purpose allocator to catch memory leaks
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

/// uses the C allocator for performance
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

const RenderContext = struct {
    glfw: GLFW,
    vulkan: Vulkan,
    current_frame: usize,
    frame_buffer_resized: bool,

    const Self = @This();

    fn init(allocator: *Allocator) !RenderContext {
        const glfw = try GLFW.init();
        errdefer glfw.deinit();

        var vulkan = try Vulkan.init(allocator, glfw.window);
        errdefer vulkan.deinit();

        return RenderContext{
            .vulkan = vulkan,
            .glfw = glfw,
            .current_frame = 0,
            .frame_buffer_resized = false,
        };
    }

    fn deinit(self: Self) void {
        self.vulkan.deinit();
        self.glfw.deinit();
    }

    fn renderFrame(self: *Self) !void {
        glfwPollEvents();
        try drawFrame(self);
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn shouldClose(self: Self) bool {
        return glfwWindowShouldClose(self.glfw.window) != GLFW_FALSE;
    }

    fn drawFrame(self: *@This()) !void {
        var vulkan = &self.vulkan;
        var window = self.glfw.window;
        var current_frame = self.current_frame;
        try checkSuccess(
            vkWaitForFences(vulkan.logical_device, 1, &vulkan.sync.in_flight_fences[current_frame], VK_TRUE, MAX_UINT64),
            error.VulkanWaitForFencesFailure,
        );

        var image_index: u32 = 0;
        {
            const result = vkAcquireNextImageKHR(
                vulkan.logical_device,
                vulkan.swap_chain.swap_chain,
                MAX_UINT64,
                vulkan.sync.image_available_semaphores[current_frame],
                null,
                &image_index,
            );
            if (result == VkResult.VK_ERROR_OUT_OF_DATE_KHR) {
                // swap chain cannot be used (e.g. due to window resize)
                try vulkan.recreateSwapChain(window);
                return;
            } else if (result != VkResult.VK_SUCCESS and result != VkResult.VK_SUBOPTIMAL_KHR) {
                return error.VulkanSwapChainAcquireNextImageFailure;
            } else {
                // swap chain may be suboptimal, but we go ahead and render anyways and recreate it later
            }
        }

        // check if a previous frame is using this image (i.e. it has a fence to wait on)
        if (vulkan.sync.images_in_flight[image_index]) |fence| {
            try checkSuccess(
                vkWaitForFences(vulkan.logical_device, 1, &fence, VK_TRUE, MAX_UINT64),
                error.VulkanWaitForFenceFailure,
            );
        }
        // mark the image as now being in use by this frame
        vulkan.sync.images_in_flight[image_index] = vulkan.sync.in_flight_fences[current_frame];

        const wait_semaphores = [_]VkSemaphore{vulkan.sync.image_available_semaphores[current_frame]};
        const signal_semaphores = [_]VkSemaphore{vulkan.sync.render_finished_semaphores[current_frame]};
        const wait_stages = [_]VkPipelineStageFlags{VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const submit_info = VkSubmitInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &wait_semaphores,
            .pWaitDstStageMask = &wait_stages,
            .commandBufferCount = 1,
            .pCommandBuffers = &vulkan.command_buffers[image_index],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signal_semaphores,
        };

        try checkSuccess(
            vkResetFences(vulkan.logical_device, 1, &vulkan.sync.in_flight_fences[current_frame]),
            error.VulkanResetFencesFailure,
        );

        try checkSuccess(
            vkQueueSubmit(vulkan.graphics_queue, 1, &submit_info, vulkan.sync.in_flight_fences[current_frame]),
            error.VulkanQueueSubmitFailure,
        );

        const swap_chains = [_]VkSwapchainKHR{vulkan.swap_chain.swap_chain};
        const present_info = VkPresentInfoKHR{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = null,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &signal_semaphores,
            .swapchainCount = 1,
            .pSwapchains = &swap_chains,
            .pImageIndices = &image_index,
            .pResults = null,
        };

        {
            const result = vkQueuePresentKHR(vulkan.present_queue, &present_info);
            if (result == VkResult.VK_ERROR_OUT_OF_DATE_KHR or result == VkResult.VK_SUBOPTIMAL_KHR or frame_buffer_resized) {
                frame_buffer_resized = false;
                try vulkan.recreateSwapChain(window);
            } else if (result != VkResult.VK_SUCCESS) {
                return error.VulkanQueuePresentFailure;
            }
        }
    }
};

const GLFW = struct {
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

        _ = glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

        return GLFW{ .window = window.? };
    }

    pub fn deinit(self: *const GLFW) void {
        glfwDestroyWindow(self.window);
        glfwTerminate();
    }
};

fn framebufferResizeCallback(window: ?*GLFWwindow, width: c_int, height: c_int) callconv(.C) void {
    frame_buffer_resized = true;
}

const Vulkan = struct {
    allocator: *Allocator,
    instance: VkInstance,
    physical_device: VkPhysicalDevice,
    logical_device: VkDevice,
    graphics_queue: VkQueue,
    present_queue: VkQueue,
    queue_family_indices: QueueFamilyIndices,
    surface: VkSurfaceKHR,
    swap_chain: SwapChain,
    pipeline: Pipeline,
    render_pass: VkRenderPass,
    swap_chain_framebuffers: []VkFramebuffer,
    command_pool: VkCommandPool,
    command_buffers: []VkCommandBuffer,
    sync: VulkanSynchronization,
    debug_messenger: ?VkDebugUtilsMessengerEXT,

    // TODO: use errdefer to clean up stuff in case of errors
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

        const swap_chain = try SwapChain.init(
            allocator,
            physical_device,
            logical_device,
            window,
            surface,
            indices,
        );

        const render_pass = try createRenderPass(logical_device, swap_chain.image_format, swap_chain.extent);
        const pipeline = try Pipeline.init(logical_device, render_pass, swap_chain.extent);

        const swap_chain_framebuffers = try createFramebuffers(allocator, logical_device, render_pass, swap_chain);

        const command_pool = try createCommandPool(logical_device, indices);

        const command_buffers = try createCommandBuffers(
            allocator,
            logical_device,
            render_pass,
            command_pool,
            swap_chain_framebuffers,
            swap_chain.extent,
            pipeline.pipeline,
        );

        var sync = try VulkanSynchronization.init(allocator, logical_device, swap_chain.images.len);
        errdefer sync.deinit();

        return Vulkan{
            .allocator = allocator,
            .instance = instance,
            .physical_device = physical_device,
            .logical_device = logical_device,
            .graphics_queue = graphics_queue,
            .present_queue = present_queue,
            .queue_family_indices = indices,
            .surface = surface,
            .swap_chain = swap_chain,
            .pipeline = pipeline,
            .render_pass = render_pass,
            .swap_chain_framebuffers = swap_chain_framebuffers,
            .command_pool = command_pool,
            .command_buffers = command_buffers,
            .sync = sync,
            .debug_messenger = debug_messenger,
        };
    }

    pub fn deinit(self: *const Vulkan) void {
        const result = vkDeviceWaitIdle(self.logical_device);
        if (result != VkResult.VK_SUCCESS) {
            log.warn("Unable to wait for Vulkan device to be idle before cleanup", .{});
        }

        self.cleanUpSwapChain();

        self.sync.deinit(self.logical_device);

        vkDestroyCommandPool(self.logical_device, self.command_pool, null);
        vkDestroyDevice(self.logical_device, null);
        if (self.debug_messenger) |messenger| {
            dbg.deinitDebugMessenger(self.instance, messenger);
        }
        vkDestroySurfaceKHR(self.instance, self.surface, null);
        vkDestroyInstance(self.instance, null);
    }

    fn cleanUpSwapChain(self: *const Vulkan) void {
        const device = self.logical_device;

        for (self.swap_chain_framebuffers) |framebuffer| {
            vkDestroyFramebuffer(device, framebuffer, null);
        }
        self.allocator.free(self.swap_chain_framebuffers);

        vkFreeCommandBuffers(
            device,
            self.command_pool,
            @intCast(u32, self.command_buffers.len),
            self.command_buffers.ptr,
        );
        self.allocator.free(self.command_buffers);

        self.pipeline.deinit(device);
        vkDestroyRenderPass(device, self.render_pass, null);
        self.swap_chain.deinit(device);
    }

    fn recreateSwapChain(self: *Vulkan, window: *GLFWwindow) !void {
        try checkSuccess(vkDeviceWaitIdle(self.logical_device), error.VulkanDeviceWaitIdleFailure);

        self.cleanUpSwapChain();

        self.swap_chain = try SwapChain.init(
            self.allocator,
            self.physical_device,
            self.logical_device,
            window,
            self.surface,
            self.queue_family_indices,
        );

        self.render_pass = try createRenderPass(
            self.logical_device,
            self.swap_chain.image_format,
            self.swap_chain.extent,
        );

        self.pipeline = try Pipeline.init(
            self.logical_device,
            self.render_pass,
            self.swap_chain.extent,
        );

        self.swap_chain_framebuffers = try createFramebuffers(
            self.allocator,
            self.logical_device,
            self.render_pass,
            self.swap_chain,
        );

        self.command_buffers = try createCommandBuffers(
            self.allocator,
            self.logical_device,
            self.render_pass,
            self.command_pool,
            self.swap_chain_framebuffers,
            self.swap_chain.extent,
            self.pipeline.pipeline,
        );
    }
};

/// caller must free returned memory
fn getRequiredExtensions(allocator: *Allocator) ![][*:0]const u8 {
    var glfw_extension_count: u32 = 0;
    const glfw_extensions = @ptrCast(
        [*]const [*:0]const u8,
        glfwGetRequiredInstanceExtensions(&glfw_extension_count),
    );

    var extensions = ArrayList([*:0]const u8).init(allocator);
    errdefer extensions.deinit();

    try extensions.appendSlice(glfw_extensions[0..glfw_extension_count]);

    if (enable_validation_layers) {
        try extensions.append(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions.toOwnedSlice();
}

fn pickPhysicalDevice(
    allocator: *Allocator,
    instance: VkInstance,
    surface: VkSurfaceKHR,
) !VkPhysicalDevice {
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

fn createLogicalDevice(
    allocator: *Allocator,
    physical_device: VkPhysicalDevice,
    indices: QueueFamilyIndices,
) !VkDevice {
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

fn createShaderModule(device: VkDevice, code: []align(@alignOf(u32)) const u8) !VkShaderModule {
    const create_info = VkShaderModuleCreateInfo{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .codeSize = code.len,
        .pCode = mem.bytesAsSlice(u32, code).ptr,
    };

    var shader_module: VkShaderModule = undefined;
    try checkSuccess(
        vkCreateShaderModule(device, &create_info, null, &shader_module),
        error.VulkanShaderCreationFailed,
    );

    return shader_module;
}

const Pipeline = struct {
    layout: VkPipelineLayout,
    pipeline: VkPipeline,

    fn init(device: VkDevice, render_pass: VkRenderPass, swap_chain_extent: VkExtent2D) !@This() {
        const vert_code align(4) = @embedFile("../vert.spv").*;
        const frag_code align(4) = @embedFile("../frag.spv").*;

        const vert_module = try createShaderModule(device, &vert_code);
        defer vkDestroyShaderModule(device, vert_module, null);

        const frag_module = try createShaderModule(device, &frag_code);
        defer vkDestroyShaderModule(device, frag_module, null);

        const vert_stage_info = VkPipelineShaderStageCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = VkShaderStageFlagBits.VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert_module,
            .pName = "main",
            .pSpecializationInfo = null,
        };

        const frag_stage_info = VkPipelineShaderStageCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = VkShaderStageFlagBits.VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag_module,
            .pName = "main",
            .pSpecializationInfo = null,
        };

        const shader_stages = [_]VkPipelineShaderStageCreateInfo{ vert_stage_info, frag_stage_info };

        const vertex_input_info = VkPipelineVertexInputStateCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .vertexBindingDescriptionCount = 0,
            .pVertexBindingDescriptions = null,
            .vertexAttributeDescriptionCount = 0,
            .pVertexAttributeDescriptions = null,
        };

        const input_assembly_info = VkPipelineInputAssemblyStateCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .topology = VkPrimitiveTopology.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };

        const viewport = VkViewport{
            .x = 0.0,
            .y = 0.0,
            .width = @intToFloat(f32, swap_chain_extent.width),
            .height = @intToFloat(f32, swap_chain_extent.height),
            .minDepth = 0.0,
            .maxDepth = 1.0,
        };

        const scissor = VkRect2D{
            .offset = VkOffset2D{ .x = 0, .y = 0 },
            .extent = swap_chain_extent,
        };

        const viewport_state = VkPipelineViewportStateCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };

        const rasterizer = VkPipelineRasterizationStateCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VkPolygonMode.VK_POLYGON_MODE_FILL,
            .lineWidth = 1.0,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VkFrontFace.VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0,
            .depthBiasClamp = 0.0,
            .depthBiasSlopeFactor = 0.0,
        };

        const multisampling = VkPipelineMultisampleStateCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .sampleShadingEnable = VK_FALSE,
            .rasterizationSamples = VkSampleCountFlagBits.VK_SAMPLE_COUNT_1_BIT,
            .minSampleShading = 1.0,
            .pSampleMask = null,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE,
        };

        const color_blend_attachment = VkPipelineColorBlendAttachmentState{
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT,
            .blendEnable = VK_TRUE,
            .srcColorBlendFactor = VkBlendFactor.VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VkBlendFactor.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = VkBlendOp.VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VkBlendFactor.VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VkBlendFactor.VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VkBlendOp.VK_BLEND_OP_ADD,
        };

        const color_blending = VkPipelineColorBlendStateCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .logicOpEnable = VK_FALSE,
            .logicOp = VkLogicOp.VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment,
            .blendConstants = [_]f32{ 0.0, 0.0, 0.0, 0.0 },
        };

        const dynamic_states = [_]VkDynamicState{
            VkDynamicState.VK_DYNAMIC_STATE_VIEWPORT,
            VkDynamicState.VK_DYNAMIC_STATE_LINE_WIDTH,
        };
        const dynamic_state = VkPipelineDynamicStateCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .dynamicStateCount = 2,
            .pDynamicStates = &dynamic_states,
        };

        const pipeline_layout_info = VkPipelineLayoutCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .setLayoutCount = 0,
            .pSetLayouts = null,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = null,
        };
        var pipeline_layout: VkPipelineLayout = undefined;
        try checkSuccess(
            vkCreatePipelineLayout(device, &pipeline_layout_info, null, &pipeline_layout),
            error.VulkanPipelineLayoutCreationFailed,
        );

        const pipeline_info = VkGraphicsPipelineCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stageCount = 2,
            .pStages = &shader_stages,
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly_info,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = null,
            .pColorBlendState = &color_blending,
            .pDynamicState = null,
            .pTessellationState = null,
            .layout = pipeline_layout,
            .renderPass = render_pass,
            .subpass = 0,
            .basePipelineHandle = null,
            .basePipelineIndex = -1,
        };
        var pipeline: VkPipeline = undefined;
        try checkSuccess(
            vkCreateGraphicsPipelines(device, null, 1, &pipeline_info, null, &pipeline),
            error.VulkanPipelineCreationFailed,
        );

        return Pipeline{
            .layout = pipeline_layout,
            .pipeline = pipeline,
        };
    }

    fn deinit(self: @This(), device: VkDevice) void {
        vkDestroyPipeline(device, self.pipeline, null);
        vkDestroyPipelineLayout(device, self.layout, null);
    }
};

fn createRenderPass(
    device: VkDevice,
    swap_chain_image_format: VkFormat,
    swap_chain_extent: VkExtent2D,
) !VkRenderPass {
    const color_attachment = VkAttachmentDescription{
        .flags = 0,
        .format = swap_chain_image_format,
        .samples = VkSampleCountFlagBits.VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VkImageLayout.VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VkImageLayout.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    const color_attachment_ref = VkAttachmentReference{
        .attachment = 0,
        .layout = VkImageLayout.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    const subpass = VkSubpassDescription{
        .flags = 0,
        .pipelineBindPoint = VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
        .inputAttachmentCount = 0,
        .pInputAttachments = null,
        .pResolveAttachments = null,
        .pDepthStencilAttachment = null,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = null,
    };

    const dependency = VkSubpassDependency{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = 0,
    };

    var render_pass: VkRenderPass = undefined;
    const render_pass_info = VkRenderPassCreateInfo{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };
    try checkSuccess(
        vkCreateRenderPass(device, &render_pass_info, null, &render_pass),
        error.VulkanRenderPassCreationFailed,
    );

    return render_pass;
}

fn createFramebuffers(
    allocator: *Allocator,
    device: VkDevice,
    render_pass: VkRenderPass,
    swap_chain: SwapChain,
) ![]VkFramebuffer {
    var framebuffers = try allocator.alloc(VkFramebuffer, swap_chain.image_views.len);
    errdefer allocator.free(framebuffers);

    for (swap_chain.image_views) |image_view, i| {
        var attachments = [_]VkImageView{image_view};
        const frame_buffer_info = VkFramebufferCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = &attachments,
            .width = swap_chain.extent.width,
            .height = swap_chain.extent.height,
            .layers = 1,
        };

        try checkSuccess(
            vkCreateFramebuffer(device, &frame_buffer_info, null, &framebuffers[i]),
            error.VulkanFramebufferCreationFailed,
        );
    }

    return framebuffers;
}

fn createCommandPool(device: VkDevice, indices: QueueFamilyIndices) !VkCommandPool {
    const pool_info = VkCommandPoolCreateInfo{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .queueFamilyIndex = indices.graphics_family.?,
    };

    var command_pool: VkCommandPool = undefined;
    try checkSuccess(
        vkCreateCommandPool(device, &pool_info, null, &command_pool),
        error.VulkanCommandPoolCreationFailure,
    );

    return command_pool;
}

fn createCommandBuffers(
    allocator: *Allocator,
    device: VkDevice,
    render_pass: VkRenderPass,
    command_pool: VkCommandPool,
    framebuffers: []VkFramebuffer,
    swap_chain_extent: VkExtent2D,
    graphics_pipeline: VkPipeline,
) ![]VkCommandBuffer {
    var buffers = try allocator.alloc(VkCommandBuffer, framebuffers.len);
    errdefer allocator.free(buffers);

    const alloc_info = VkCommandBufferAllocateInfo{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = null,
        .commandPool = command_pool,
        .level = VkCommandBufferLevel.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = @intCast(u32, buffers.len),
    };

    try checkSuccess(
        vkAllocateCommandBuffers(device, &alloc_info, buffers.ptr),
        error.VulkanCommanbBufferAllocationFailure,
    );

    for (buffers) |buffer, i| {
        const begin_info = VkCommandBufferBeginInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = 0,
            .pInheritanceInfo = null,
        };
        try checkSuccess(
            vkBeginCommandBuffer(buffer, &begin_info),
            error.VulkanBeginCommandBufferFailure,
        );

        const clear_color = [_]VkClearValue{VkClearValue{
            .color = VkClearColorValue{ .float32 = [_]f32{ 0.0, 0.0, 0.0, 1.0 } },
        }};
        const render_pass_info = VkRenderPassBeginInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = null,
            .renderPass = render_pass,
            .framebuffer = framebuffers[i],
            .renderArea = VkRect2D{
                .offset = VkOffset2D{ .x = 0, .y = 0 },
                .extent = swap_chain_extent,
            },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
        };

        vkCmdBeginRenderPass(buffer, &render_pass_info, VkSubpassContents.VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(buffer, VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);
        vkCmdDraw(buffer, 3, 1, 0, 0);
        vkCmdEndRenderPass(buffer);

        try checkSuccess(
            vkEndCommandBuffer(buffer),
            error.VulkanCommandBufferEndFailure,
        );
    }

    return buffers;
}

const VulkanSynchronization = struct {
    const Self = @This();

    allocator: *Allocator,
    image_available_semaphores: []VkSemaphore,
    render_finished_semaphores: []VkSemaphore,
    in_flight_fences: []VkFence,
    images_in_flight: []?VkFence,

    fn init(allocator: *Allocator, device: VkDevice, image_count: usize) !Self {
        var image_available_semaphores = try allocator.alloc(VkSemaphore, MAX_FRAMES_IN_FLIGHT);
        errdefer allocator.free(image_available_semaphores);

        var render_finished_semaphores = try allocator.alloc(VkSemaphore, MAX_FRAMES_IN_FLIGHT);
        errdefer allocator.free(render_finished_semaphores);

        var in_flight_fences = try allocator.alloc(VkFence, MAX_FRAMES_IN_FLIGHT);
        errdefer allocator.free(in_flight_fences);

        var images_in_flight = try allocator.alloc(?VkFence, image_count);
        errdefer allocator.free(images_in_flight);

        var i: usize = 0;
        while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
            const semaphore = try createSemaphore(device);
            errdefer vkDestroySemaphore(device, semaphore);
            image_available_semaphores[i] = semaphore;
        }

        i = 0;
        while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
            const semaphore = try createSemaphore(device);
            errdefer vkDestroySemaphore(device, semaphore);
            render_finished_semaphores[i] = semaphore;
        }

        i = 0;
        while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
            const fence = try createFence(device);
            errdefer vkDestroyFence(device, fence, null);
            in_flight_fences[i] = fence;
        }

        i = 0;
        while (i < image_count) : (i += 1) {
            images_in_flight[i] = null;
        }

        return VulkanSynchronization{
            .allocator = allocator,
            .image_available_semaphores = image_available_semaphores,
            .render_finished_semaphores = render_finished_semaphores,
            .in_flight_fences = in_flight_fences,
            .images_in_flight = images_in_flight,
        };
    }

    fn deinit(self: Self, device: VkDevice) void {
        for (self.render_finished_semaphores) |semaphore| {
            vkDestroySemaphore(device, semaphore, null);
        }
        self.allocator.free(self.render_finished_semaphores);

        for (self.image_available_semaphores) |semaphore| {
            vkDestroySemaphore(device, semaphore, null);
        }
        self.allocator.free(self.image_available_semaphores);

        for (self.in_flight_fences) |fence| {
            vkDestroyFence(device, fence, null);
        }
        self.allocator.free(self.in_flight_fences);

        self.allocator.free(self.images_in_flight);
    }
};

fn createSemaphore(device: VkDevice) !VkSemaphore {
    var semaphore: VkSemaphore = undefined;
    const semaphore_info = VkSemaphoreCreateInfo{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
    };
    try checkSuccess(
        vkCreateSemaphore(device, &semaphore_info, null, &semaphore),
        error.VulkanSemaphoreCreationFailure,
    );

    return semaphore;
}

fn createFence(device: VkDevice) !VkFence {
    var fence: VkFence = undefined;
    const info = VkFenceCreateInfo{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = null,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    try checkSuccess(
        vkCreateFence(device, &info, null, &fence),
        error.VulkanFenceCreationFailed,
    );

    return fence;
}
