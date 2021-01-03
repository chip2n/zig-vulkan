pub usingnamespace @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", {});
    @cInclude("GLFW/glfw3.h");
    @cInclude("vulkan/vulkan.h");
});

pub const glm = @cImport({
    @cInclude("cglm/cglm.h");
});

pub const vec2 = glm.vec2;
pub const vec3 = glm.vec3;
