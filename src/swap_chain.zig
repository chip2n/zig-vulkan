const std = @import("std");
const Allocator = std.mem.Allocator;
const log = std.log;

usingnamespace @import("c.zig");
usingnamespace @import("queue_family.zig");
usingnamespace @import("utils.zig");

pub const SwapChain = struct {
    allocator: *Allocator,
    swap_chain: VkSwapchainKHR,
    swap_chain_images: []VkImage,
    swap_chain_image_format: VkFormat,
    swap_chain_extent: VkExtent2D,

    pub fn init(
        allocator: *Allocator,
        physical_device: VkPhysicalDevice,
        logical_device: VkDevice,
        window: *GLFWwindow,
        surface: VkSurfaceKHR,
        indices: QueueFamilyIndices,
    ) !SwapChain {
        const swap_chain = try createSwapChain(allocator, physical_device, logical_device, window, surface, indices);
        var image_count: u32 = 0;
        try checkSuccess(
            vkGetSwapchainImagesKHR(logical_device, swap_chain, &image_count, null),
            error.VulkanSwapChainImageRetrievalFailed,
        );
        var swap_chain_images = try allocator.alloc(VkImage, image_count);
        try checkSuccess(
            vkGetSwapchainImagesKHR(logical_device, swap_chain, &image_count, swap_chain_images.ptr),
            error.VulkanSwapChainImageRetrievalFailed,
        );
        // TODO reuse this
        const swap_chain_support = try querySwapChainSupport(allocator, physical_device, surface);
        defer swap_chain_support.deinit();
        const swap_chain_surface_format = chooseSwapSurfaceFormat(swap_chain_support.formats).format;
        const swap_chain_extent = chooseSwapExtent(window, swap_chain_support.capabilities);

        return SwapChain{
            .allocator = allocator,
            .swap_chain = swap_chain,
            .swap_chain_images = swap_chain_images,
            .swap_chain_image_format = swap_chain_surface_format,
            .swap_chain_extent = swap_chain_extent,
        };
    }

    pub fn deinit(self: *const SwapChain, logical_device: VkDevice) void {
        self.allocator.free(self.swap_chain_images);
        vkDestroySwapchainKHR(logical_device, self.swap_chain, null);
    }
};

const SwapChainSupportDetails = struct {
    allocator: *Allocator,
    capabilities: VkSurfaceCapabilitiesKHR,
    formats: []VkSurfaceFormatKHR,
    present_modes: []VkPresentModeKHR,

    fn init(
        allocator: *Allocator,
        capabilities: VkSurfaceCapabilitiesKHR,
        formats: []VkSurfaceFormatKHR,
        present_modes: []VkPresentModeKHR,
    ) SwapChainSupportDetails {
        return .{
            .allocator = allocator,
            .capabilities = capabilities,
            .formats = formats,
            .present_modes = present_modes,
        };
    }

    pub fn deinit(self: @This()) void {
        self.allocator.free(self.formats);
        self.allocator.free(self.present_modes);
    }
};

pub fn querySwapChainSupport(allocator: *Allocator, device: VkPhysicalDevice, surface: VkSurfaceKHR) !SwapChainSupportDetails {
    var capabilities: VkSurfaceCapabilitiesKHR = undefined;
    try checkSuccess(
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &capabilities),
        error.VulkanSurfaceCapabilitiesQueryFailed,
    );

    var format_count: u32 = 0;
    try checkSuccess(
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, null),
        error.VulkanSurfaceFormatsQueryFailed,
    );
    var formats = try allocator.alloc(VkSurfaceFormatKHR, format_count);
    errdefer allocator.free(formats);
    if (format_count != 0) {
        try checkSuccess(
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, formats.ptr),
            error.VulkanSurfaceFormatsQueryFailed,
        );
    }

    var present_mode_count: u32 = 0;
    try checkSuccess(
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, null),
        error.VulkanSurfacePresentModesQueryFailed,
    );
    var present_modes = try allocator.alloc(VkPresentModeKHR, present_mode_count);
    errdefer allocator.free(present_modes);
    if (present_mode_count != 0) {
        try checkSuccess(
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, present_modes.ptr),
            error.VulkanSurfacePresentModesQueryFailed,
        );
    }

    return SwapChainSupportDetails.init(
        allocator,
        capabilities,
        formats,
        present_modes,
    );
}

pub fn chooseSwapSurfaceFormat(available_formats: []VkSurfaceFormatKHR) VkSurfaceFormatKHR {
    // Try to find SRGB format
    for (available_formats) |format| {
        if (format.format == VkFormat.VK_FORMAT_B8G8R8A8_SRGB and format.colorSpace == VkColorSpaceKHR.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    log.warn("SRGB format not found - picking the first available format", .{});
    return available_formats[0];
}

pub fn chooseSwapPresentMode(available_present_modes: []VkPresentModeKHR) VkPresentModeKHR {
    for (available_present_modes) |present_mode| {
        if (present_mode == VkPresentModeKHR.VK_PRESENT_MODE_MAILBOX_KHR) {
            log.info("using vulkan present mode: VK_PRESENT_MODE_MAILBOX_KHR", .{});
            return present_mode;
        }
    }

    log.info("using vulkan present mode: VK_PRESENT_MODE_FIFO_KHR", .{});
    return VkPresentModeKHR.VK_PRESENT_MODE_FIFO_KHR;
}

pub fn chooseSwapExtent(window: *GLFWwindow, capabilities: VkSurfaceCapabilitiesKHR) VkExtent2D {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        var width: c_int = 0;
        var height: c_int = 0;
        glfwGetFramebufferSize(window, &width, &height);

        var actual_extent = VkExtent2D{ .width = @intCast(u32, width), .height = @intCast(u32, height) };
        actual_extent.width = std.math.max(
            capabilities.minImageExtent.width,
            std.math.min(capabilities.maxImageExtent.width, actual_extent.width),
        );
        actual_extent.height = std.math.max(
            capabilities.minImageExtent.height,
            std.math.min(capabilities.maxImageExtent.height, actual_extent.height),
        );

        return actual_extent;
    }
}

pub fn createSwapChain(
    allocator: *Allocator,
    physical_device: VkPhysicalDevice,
    logical_device: VkDevice,
    window: *GLFWwindow,
    surface: VkSurfaceKHR,
    indices: QueueFamilyIndices,
) !VkSwapchainKHR {
    const swap_chain_support = try querySwapChainSupport(allocator, physical_device, surface);
    defer swap_chain_support.deinit();
    const surface_format = chooseSwapSurfaceFormat(swap_chain_support.formats);
    const present_mode = chooseSwapPresentMode(swap_chain_support.present_modes);
    const extent = chooseSwapExtent(window, swap_chain_support.capabilities);

    var image_count = swap_chain_support.capabilities.minImageCount + 1;
    if (swap_chain_support.capabilities.maxImageCount > 0 and image_count > swap_chain_support.capabilities.maxImageCount) {
        image_count = swap_chain_support.capabilities.maxImageCount;
    }

    const queue_family_indices = [_]u32{ indices.graphics_family.?, indices.present_family.? };

    var create_info = VkSwapchainCreateInfoKHR{
        .sType = VkStructureType.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = null,
        .flags = 0,
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,

        .imageSharingMode = VkSharingMode.VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = null,

        .preTransform = swap_chain_support.capabilities.currentTransform,
        .compositeAlpha = VkCompositeAlphaFlagBitsKHR.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = present_mode,
        .clipped = VK_TRUE,
        .oldSwapchain = null,
    };

    if (indices.graphics_family.? != indices.present_family.?) {
        create_info.imageSharingMode = VkSharingMode.VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = 2;
        create_info.pQueueFamilyIndices = &queue_family_indices;
    }

    var swap_chain: VkSwapchainKHR = undefined;
    try checkSuccess(
        vkCreateSwapchainKHR(logical_device, &create_info, null, &swap_chain),
        error.VulkanSwapChainCreationFailed,
    );

    return swap_chain;
}
