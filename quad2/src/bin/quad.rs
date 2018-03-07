extern crate ash;
extern crate image;
extern crate quad;

use quad::*;

fn main() {
    let mut quad = Quad::new();
    quad.render();

    // for pipeline in graphics_pipelines {
    //     base.device.destroy_pipeline(pipeline, None);
    // }
    // base.device.destroy_pipeline_layout(pipeline_layout, None);
    // base.device.destroy_shader_module(shader_module, None);
    // base.device.free_memory(index_buffer_memory, None);
    // base.device.destroy_buffer(index_buffer, None);
    // // base.device.free_memory(uniform_color_buffer_memory, None);
    // // base.device.destroy_buffer(uniform_color_buffer, None);
    // base.device.free_memory(vertex_input_buffer_memory, None);
    // base.device.destroy_buffer(vertex_input_buffer, None);
    // for &descriptor_set_layout in desc_set_layouts.iter() {
    //     base.device
    //         .destroy_descriptor_set_layout(descriptor_set_layout, None);
    // }
    // base.device.destroy_descriptor_pool(descriptor_pool, None);
    // for framebuffer in framebuffers {
    //     base.device.destroy_framebuffer(framebuffer, None);
    // }
    // base.device.destroy_render_pass(renderpass, None);
}
