const std = @import("std");
const mem = std.mem;
const builtin = @import("builtin");
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const log = std.log;
const dbg = @import("debug.zig");
const vk = @import("vulkan.zig");

usingnamespace @import("c.zig");
usingnamespace @import("queue_family.zig");
usingnamespace @import("swap_chain.zig");
usingnamespace @import("utils.zig");
usingnamespace @import("window.zig");

pub const log_level: std.log.Level = .warn;

const MAX_FRAMES_IN_FLIGHT = 2;
const MAX_UINT64 = @as(c_ulong, 18446744073709551615); // Couldn't use UINT64_MAX for some reason

const enable_validation_layers = std.debug.runtime_safety;

const device_extensions = [_][*:0]const u8{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

const Vertex = struct {
    pos: vec2,
    color: vec3,

    fn getBindingDescription() VkVertexInputBindingDescription {
        return VkVertexInputBindingDescription{
            .binding = 0,
            .stride = @sizeOf(Vertex),
            .inputRate = VkVertexInputRate.VK_VERTEX_INPUT_RATE_VERTEX,
        };
    }

    fn getAttributeDescriptions() [2]VkVertexInputAttributeDescription {
        return [2]VkVertexInputAttributeDescription{
            VkVertexInputAttributeDescription{
                .binding = 0,
                .location = 0,
                .format = VkFormat.VK_FORMAT_R32G32_SFLOAT,
                .offset = @byteOffsetOf(Vertex, "pos"),
            },
            VkVertexInputAttributeDescription{
                .binding = 0,
                .location = 1,
                .format = VkFormat.VK_FORMAT_R32G32B32_SFLOAT,
                .offset = @byteOffsetOf(Vertex, "color"),
            },
        };
    }
};

const vertices = [_]Vertex{
    Vertex{ .pos = vec2{ 0.0, -0.5 }, .color = vec3{ 1.0, 0.0, 0.0 } },
    Vertex{ .pos = vec2{ 0.5, 0.5 }, .color = vec3{ 0.0, 1.0, 0.0 } },
    Vertex{ .pos = vec2{ -0.5, 0.5 }, .color = vec3{ 0.0, 0.0, 1.0 } },
};

pub fn main() !void {
    var system = System().init();
    defer system.deinit();

    const allocator = system.allocator();

    var context = try RenderContext.init(allocator);
    defer context.deinit();

    var callback = ResizeCallback{
        .data = &context,
        .cb = framebufferResizeCallback,
    };
    context.window.registerResizeCallback(&callback);

    while (!context.shouldClose()) {
        try context.renderFrame();
    }
}

fn framebufferResizeCallback(data: *c_void) void {
    var context = @ptrCast(*RenderContext, @alignCast(@alignOf(*RenderContext), data));
    context.framebuffer_resized = true;
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
    window: Window,
    vulkan: Vulkan,
    current_frame: usize,
    framebuffer_resized: bool,

    const Self = @This();

    fn init(allocator: *Allocator) !RenderContext {
        const window = try Window.init();
        errdefer window.deinit();

        var vulkan = try Vulkan.init(allocator, &window);
        errdefer vulkan.deinit();

        return RenderContext{
            .vulkan = vulkan,
            .window = window,
            .current_frame = 0,
            .framebuffer_resized = false,
        };
    }

    fn deinit(self: Self) void {
        self.vulkan.deinit();
        self.window.deinit();
    }

    fn renderFrame(self: *Self) !void {
        self.window.pollEvents();
        try drawFrame(self);
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn shouldClose(self: Self) bool {
        return self.window.shouldClose();
    }

    fn drawFrame(self: *@This()) !void {
        var vulkan = &self.vulkan;
        var window = &self.window;
        var current_frame = self.current_frame;
        try checkSuccess(
            vkWaitForFences(vulkan.device, 1, &vulkan.sync.in_flight_fences[current_frame], VK_TRUE, MAX_UINT64),
            error.VulkanWaitForFencesFailure,
        );

        var image_index: u32 = 0;
        {
            const result = vkAcquireNextImageKHR(
                vulkan.device,
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
                vkWaitForFences(vulkan.device, 1, &fence, VK_TRUE, MAX_UINT64),
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
            vkResetFences(vulkan.device, 1, &vulkan.sync.in_flight_fences[current_frame]),
            error.VulkanResetFencesFailure,
        );

        try vk.queueSubmit(vulkan.graphics_queue, 1, &submit_info, vulkan.sync.in_flight_fences[current_frame]);

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
            if (result == VkResult.VK_ERROR_OUT_OF_DATE_KHR or result == VkResult.VK_SUBOPTIMAL_KHR or self.framebuffer_resized) {
                self.framebuffer_resized = false;
                try vulkan.recreateSwapChain(window);
            } else if (result != VkResult.VK_SUCCESS) {
                return error.VulkanQueuePresentFailure;
            }
        }
    }
};

const Vulkan = struct {
    allocator: *Allocator,
    instance: VkInstance,
    physical_device: VkPhysicalDevice,
    device: VkDevice,
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
    vertex_buffer: VertexBuffer,
    debug_messenger: ?VkDebugUtilsMessengerEXT,

    // TODO: use errdefer to clean up stuff in case of errors
    pub fn init(allocator: *Allocator, window: *const Window) !Vulkan {
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

        const surface = try window.createSurface(instance);

        const physical_device = try pickPhysicalDevice(allocator, instance, surface);
        const indices = try findQueueFamilies(allocator, physical_device, surface);
        if (!indices.isComplete()) {
            return error.VulkanSuitableQueuFamiliesNotFound;
        }

        const device = try createLogicalDevice(allocator, physical_device, indices);

        var graphics_queue: VkQueue = undefined;
        vkGetDeviceQueue(
            device,
            indices.graphics_family.?,
            0,
            &graphics_queue,
        );

        var present_queue: VkQueue = undefined;
        vkGetDeviceQueue(
            device,
            indices.present_family.?,
            0,
            &present_queue,
        );

        const swap_chain = try SwapChain.init(
            allocator,
            physical_device,
            device,
            window,
            surface,
            indices,
        );

        const render_pass = try createRenderPass(device, swap_chain.image_format, swap_chain.extent);
        const pipeline = try Pipeline.init(device, render_pass, swap_chain.extent);

        const swap_chain_framebuffers = try createFramebuffers(allocator, device, render_pass, swap_chain);

        const command_pool = try createCommandPool(device, indices);

        const vertex_buffer = try VertexBuffer.init(physical_device, device, graphics_queue, command_pool);

        const command_buffers = try createCommandBuffers(
            allocator,
            device,
            render_pass,
            command_pool,
            swap_chain_framebuffers,
            swap_chain.extent,
            pipeline.pipeline,
            vertex_buffer.buffer,
        );

        var sync = try VulkanSynchronization.init(allocator, device, swap_chain.images.len);
        errdefer sync.deinit(device);

        return Vulkan{
            .allocator = allocator,
            .instance = instance,
            .physical_device = physical_device,
            .device = device,
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
            .vertex_buffer = vertex_buffer,
            .debug_messenger = debug_messenger,
        };
    }

    pub fn deinit(self: *const Vulkan) void {
        const result = vkDeviceWaitIdle(self.device);
        if (result != VkResult.VK_SUCCESS) {
            log.warn("Unable to wait for Vulkan device to be idle before cleanup", .{});
        }

        self.cleanUpSwapChain();

        self.vertex_buffer.deinit(self.device);

        self.sync.deinit(self.device);

        vkDestroyCommandPool(self.device, self.command_pool, null);
        vkDestroyDevice(self.device, null);
        if (self.debug_messenger) |messenger| {
            dbg.deinitDebugMessenger(self.instance, messenger);
        }
        vkDestroySurfaceKHR(self.instance, self.surface, null);
        vkDestroyInstance(self.instance, null);
    }

    fn cleanUpSwapChain(self: *const Vulkan) void {
        for (self.swap_chain_framebuffers) |framebuffer| {
            vkDestroyFramebuffer(self.device, framebuffer, null);
        }
        self.allocator.free(self.swap_chain_framebuffers);

        vkFreeCommandBuffers(
            self.device,
            self.command_pool,
            @intCast(u32, self.command_buffers.len),
            self.command_buffers.ptr,
        );
        self.allocator.free(self.command_buffers);

        self.pipeline.deinit(self.device);
        vkDestroyRenderPass(self.device, self.render_pass, null);
        self.swap_chain.deinit(self.device);
    }

    fn recreateSwapChain(self: *Vulkan, window: *const Window) !void {
        while (window.isMinimized()) {
            window.waitEvents();
        }

        try checkSuccess(vkDeviceWaitIdle(self.device), error.VulkanDeviceWaitIdleFailure);

        self.cleanUpSwapChain();

        self.swap_chain = try SwapChain.init(
            self.allocator,
            self.physical_device,
            self.device,
            window,
            self.surface,
            self.queue_family_indices,
        );

        self.render_pass = try createRenderPass(
            self.device,
            self.swap_chain.image_format,
            self.swap_chain.extent,
        );

        self.pipeline = try Pipeline.init(
            self.device,
            self.render_pass,
            self.swap_chain.extent,
        );

        self.swap_chain_framebuffers = try createFramebuffers(
            self.allocator,
            self.device,
            self.render_pass,
            self.swap_chain,
        );

        self.command_buffers = try createCommandBuffers(
            self.allocator,
            self.device,
            self.render_pass,
            self.command_pool,
            self.swap_chain_framebuffers,
            self.swap_chain.extent,
            self.pipeline.pipeline,
            self.vertex_buffer.buffer,
        );
    }
};

/// caller must free returned memory
fn getRequiredExtensions(allocator: *Allocator) ![][*:0]const u8 {
    var extensions = try getWindowRequiredExtensions(allocator);
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

        const binding_desc = Vertex.getBindingDescription();
        const attr_descs = Vertex.getAttributeDescriptions();
        const vertex_input_info = VkPipelineVertexInputStateCreateInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &binding_desc,
            .vertexAttributeDescriptionCount = attr_descs.len,
            .pVertexAttributeDescriptions = &attr_descs,
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
    vertex_buffer: VkBuffer,
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

    try vk.allocateCommandBuffers(device, &alloc_info, buffers.ptr);

    for (buffers) |buffer, i| {
        const begin_info = VkCommandBufferBeginInfo{
            .sType = VkStructureType.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = 0,
            .pInheritanceInfo = null,
        };
        try vk.beginCommandBuffer(buffer, &begin_info);

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

        const vertex_buffers = [_]VkBuffer{vertex_buffer};
        const offsets = [_]VkDeviceSize{0};
        vkCmdBindVertexBuffers(buffer, 0, 1, &vertex_buffers, &offsets);
        vkCmdDraw(buffer, vertices.len, 1, 0, 0);
        vkCmdEndRenderPass(buffer);

        try vk.endCommandBuffer(buffer);
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

    try checkSuccess(vkCreateFence(device, &info, null, &fence), error.VulkanFenceCreationFailed);

    return fence;
}

const VertexBuffer = struct {
    const Self = @This();

    buffer: VkBuffer,
    memory: VkDeviceMemory,

    fn init(
        physical_device: VkPhysicalDevice,
        device: VkDevice,
        graphics_queue: VkQueue,
        command_pool: VkCommandPool,
    ) !Self {
        const buffer_size = @sizeOf(Vertex) * vertices.len;

        var staging_buffer: VkBuffer = undefined;
        var staging_memory: VkDeviceMemory = undefined;
        try createBuffer(
            physical_device,
            device,
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &staging_buffer,
            &staging_memory,
        );

        var data: ?*c_void = undefined;
        try checkSuccess(vkMapMemory(device, staging_memory, 0, buffer_size, 0, &data), error.VulkanMapMemoryError);
        @memcpy(@ptrCast([*]u8, data), @ptrCast([*]align(4) const u8, &vertices), buffer_size);
        vkUnmapMemory(device, staging_memory);

        var vertex_buffer: VkBuffer = undefined;
        var vertex_memory: VkDeviceMemory = undefined;
        try createBuffer(
            physical_device,
            device,
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &vertex_buffer,
            &vertex_memory,
        );

        try copyBuffer(device, graphics_queue, command_pool, staging_buffer, vertex_buffer, buffer_size);

        vkDestroyBuffer(device, staging_buffer, null);
        vkFreeMemory(device, staging_memory, null);

        return Self{ .buffer = vertex_buffer, .memory = vertex_memory };
    }

    fn deinit(self: Self, device: VkDevice) void {
        vkDestroyBuffer(device, self.buffer, null);
        vkFreeMemory(device, self.memory, null);
    }
};

fn copyBuffer(
    device: VkDevice,
    graphics_queue: VkQueue,
    command_pool: VkCommandPool,
    src: VkBuffer,
    dst: VkBuffer,
    size: VkDeviceSize,
) !void {
    // OPTIMIZE: Create separate command pool for short lived buffers
    const alloc_info = VkCommandBufferAllocateInfo{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = null,
        .level = VkCommandBufferLevel.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandPool = command_pool,
        .commandBufferCount = 1,
    };
    var command_buffer: VkCommandBuffer = undefined;
    try vk.allocateCommandBuffers(device, &alloc_info, &command_buffer);
    defer vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);

    const begin_info = VkCommandBufferBeginInfo{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = null,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = null,
    };

    try vk.beginCommandBuffer(command_buffer, &begin_info);

    const copy_region = VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = size };
    vkCmdCopyBuffer(command_buffer, src, dst, 1, &copy_region);
    try vk.endCommandBuffer(command_buffer);

    const submit_info = VkSubmitInfo{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = null,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = null,
        .pWaitDstStageMask = null,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = null,
    };

    try vk.queueSubmit(graphics_queue, 1, &submit_info, null);
    try vk.queueWaitIdle(graphics_queue);
}

fn findMemoryType(physical_device: VkPhysicalDevice, type_filter: u32, properties: VkMemoryPropertyFlags) !u32 {
    var mem_props: VkPhysicalDeviceMemoryProperties = undefined;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

    var i: u32 = 0;
    while (i < mem_props.memoryTypeCount) : (i += 1) {
        if (type_filter & (@intCast(u32, 1) << @intCast(u5, i)) != 0 and
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    return error.VulkanSuitableMemoryTypeNotFound;
}

fn createBuffer(
    physical_device: VkPhysicalDevice,
    device: VkDevice,
    size: VkDeviceSize,
    usage: VkBufferUsageFlags,
    properties: VkMemoryPropertyFlags,
    buffer: *VkBuffer,
    buffer_memory: *VkDeviceMemory,
) !void {
    const info = VkBufferCreateInfo{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .size = size,
        .usage = usage,
        .sharingMode = VkSharingMode.VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = null,
    };

    try checkSuccess(vkCreateBuffer(device, &info, null, buffer), error.VulkanVertexBufferCreationFailed);

    var mem_reqs: VkMemoryRequirements = undefined;
    vkGetBufferMemoryRequirements(device, buffer.*, &mem_reqs);

    const alloc_info = VkMemoryAllocateInfo{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = null,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = try findMemoryType(physical_device, mem_reqs.memoryTypeBits, properties),
    };

    try checkSuccess(vkAllocateMemory(device, &alloc_info, null, buffer_memory), error.VulkanAllocateMemoryFailure);
    try checkSuccess(vkBindBufferMemory(device, buffer.*, buffer_memory.*, 0), error.VulkanBindBufferMemoryFailure);
}
