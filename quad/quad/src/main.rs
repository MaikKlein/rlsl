extern crate clap;
extern crate env_logger;
extern crate gfx_backend_vulkan as back;
extern crate gfx_hal as hal;
extern crate image;
#[macro_use]
extern crate structopt;
extern crate winit;

use clap::{App, Arg, SubCommand};
use std::path::Path;
use std::cell::RefCell;
use hal::{buffer, command, device as d, format as f, image as i, memory as m, pass, pool, pso};
use hal::{Backend, Device, Instance, PhysicalDevice, Surface, Swapchain};
use hal::{Backbuffer, DescriptorPool, FrameSync, Primitive, SwapchainConfig};
use hal::format::{AsFormat, ChannelType, Rgba8Srgb as ColorFormat, Swizzle};
use hal::pass::Subpass;
use hal::pso::{PipelineStage, ShaderStageFlags, Specialization};
use hal::queue::Submission;
use hal::queue::capability::Graphics;

use std::io::Cursor;
use std::ops::Range;

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
struct Vertex {
    a_Pos: [f32; 2],
    a_Uv: [f32; 2],
}

const QUAD: [Vertex; 6] = [
    Vertex {
        a_Pos: [-1.0, 1.0],
        a_Uv: [0.0, 1.0],
    },
    Vertex {
        a_Pos: [1.0, 1.0],
        a_Uv: [1.0, 1.0],
    },
    Vertex {
        a_Pos: [1.0, -1.0],
        a_Uv: [1.0, 0.0],
    },
    Vertex {
        a_Pos: [-1.0, 1.0],
        a_Uv: [0.0, 1.0],
    },
    Vertex {
        a_Pos: [1.0, -1.0],
        a_Uv: [1.0, 0.0],
    },
    Vertex {
        a_Pos: [-1.0, -1.0],
        a_Uv: [0.0, 0.0],
    },
];

const COLOR_RANGE: i::SubresourceRange = i::SubresourceRange {
    aspects: f::AspectFlags::COLOR,
    levels: 0..1,
    layers: 0..1,
};
fn create_graphics_pipeline<B: hal::Backend>(
    device: &B::Device,
    pipeline_layout: &B::PipelineLayout,
    render_pass: &B::RenderPass,
    vertex_path: &Path,
    fragment_path: &Path,
    vertex_entry: &str,
    fragment_entry: &str,
) -> B::GraphicsPipeline {
    let pipeline = {
        use std::fs::File;
        use std::io::Read;
        let spv_file_frag = File::open(fragment_path).expect("file");
        let spv_bytes_frag: Vec<u8> = spv_file_frag.bytes().filter_map(|byte| byte.ok()).collect();
        let spv_file_vert = File::open(vertex_path).expect("file");
        let spv_bytes_vert: Vec<u8> = spv_file_vert.bytes().filter_map(|byte| byte.ok()).collect();
        let spv_module_frag = device.create_shader_module(&spv_bytes_frag).unwrap();
        let spv_module_vert = device.create_shader_module(&spv_bytes_vert).unwrap();

        let pipeline = {
            let (vs_entry, fs_entry) = (
                pso::EntryPoint::<B> {
                    entry: vertex_entry,
                    module: &spv_module_vert,
                    specialization: &[
                        // Specialization {
                        //     id: 0,
                        //     value: pso::Constant::F32(0.8),
                        // }
                    ],
                },
                pso::EntryPoint::<B> {
                    entry: fragment_entry,
                    module: &spv_module_frag,
                    specialization: &[],
                },
            );

            let shader_entries = pso::GraphicsShaderSet {
                vertex: vs_entry,
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(fs_entry),
            };

            let subpass = Subpass {
                index: 0,
                main_pass: render_pass,
            };

            let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
                shader_entries,
                Primitive::TriangleList,
                pso::Rasterizer::FILL,
                &pipeline_layout,
                subpass,
            );
            pipeline_desc.blender.targets.push(pso::ColorBlendDesc(
                pso::ColorMask::ALL,
                pso::BlendState::ALPHA,
            ));
            pipeline_desc.vertex_buffers.push(pso::VertexBufferDesc {
                stride: std::mem::size_of::<Vertex>() as u32,
                rate: 0,
            });

            pipeline_desc.attributes.push(pso::AttributeDesc {
                location: 0,
                binding: 0,
                element: pso::Element {
                    format: f::Format::Rg32Float,
                    offset: 0,
                },
            });
            pipeline_desc.attributes.push(pso::AttributeDesc {
                location: 1,
                binding: 0,
                element: pso::Element {
                    format: f::Format::Rg32Float,
                    offset: 8,
                },
            });

            device.create_graphics_pipeline(&pipeline_desc)
        };

        device.destroy_shader_module(spv_module_frag);
        device.destroy_shader_module(spv_module_vert);
        pipeline
    };
    pipeline.expect("pipeline")
}

pub struct Quad<B: hal::Backend> {
    instance: back::Instance,
    device: B::Device,
    pipeline_layout: B::PipelineLayout,
    render_pass: B::RenderPass,
    command_pool: hal::pool::CommandPool<B, Graphics>,
    viewport: hal::command::Viewport,
    window: winit::Window,
    events_loop: winit::EventsLoop,
    vertex_buffer: B::Buffer,
    desc_set: B::DescriptorSet,
    swap_chain: B::Swapchain,
    framebuffers: Vec<B::Framebuffer>,
    queue: hal::queue::CommandQueue<B, Graphics>,
    desc_pool: B::DescriptorPool,
}

impl<B: hal::Backend> std::ops::Drop for Quad<B> {
    fn drop(&mut self) {
        // self.device
        //     .destroy_command_pool(self.command_pool.downgrade());
        ////device.destroy_descriptor_pool(desc_pool);
        //device.destroy_descriptor_set_layout(set_layout);

        //device.destroy_buffer(vertex_buffer);
        ////device.destroy_buffer(image_upload_buffer);
        //// device.destroy_image(image_logo);
        //// device.destroy_image_view(image_srv);
        //// device.destroy_sampler(sampler);
        //device.destroy_fence(frame_fence);
        //device.destroy_semaphore(frame_semaphore);
        //device.destroy_pipeline_layout(pipeline_layout);
        //device.destroy_renderpass(render_pass);
        //device.free_memory(buffer_memory);
        //// device.free_memory(image_memory);
        //// device.free_memory(image_upload_memory);
        //for pipeline in pipelines {
        //    device.destroy_graphics_pipeline(pipeline);
        //}
        //for framebuffer in framebuffers {
        //    device.destroy_framebuffer(framebuffer);
        //}
        //for (image, rtv) in frame_images {
        //    device.destroy_image_view(rtv);
        //    device.destroy_image(image);
        //}
    }
}

impl Quad<back::Backend> {
    pub fn new() -> Quad<back::Backend> {
        env_logger::init();

        let mut events_loop = winit::EventsLoop::new();

        let wb = winit::WindowBuilder::new()
            .with_dimensions(1024, 768)
            .with_title("quad".to_string());
        let window = wb.build(&events_loop).unwrap();

        let window_size = window.get_inner_size().unwrap();
        let pixel_width = window_size.0 as u16;
        let pixel_height = window_size.1 as u16;

        // instantiate backend
        let (_instance, mut adapters, mut surface) = {
            let instance = back::Instance::create("gfx-rs quad", 1);
            println!("TEST");
            let surface = instance.create_surface(&window);
            let adapters = instance.enumerate_adapters();
            (instance, adapters, surface)
        };
        for adapter in &adapters {
            println!("{:?}", adapter.info);
        }

        let adapter = adapters.remove(0);
        let surface_format = surface
            .capabilities_and_formats(&adapter.physical_device)
            .1
            .map_or(f::Format::Rgba8Srgb, |formats| {
                formats
                    .into_iter()
                    .find(|format| format.base_format().1 == ChannelType::Srgb)
                    .unwrap()
            });

        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.get_limits();

        // Build a new device and associated command queues
        let (device, mut queue_group) = adapter
            .open_with::<_, hal::Graphics>(1, |family| surface.supports_queue_family(family))
            .unwrap();

        let mut command_pool = device.create_command_pool_typed(
            &queue_group,
            pool::CommandPoolCreateFlags::empty(),
            16,
        );
        let mut queue = queue_group.queues.swap_remove(0);

        println!("Surface format: {:?}", surface_format);
        let swap_config = SwapchainConfig::new().with_color(surface_format);
        let (mut swap_chain, backbuffer) = device.create_swapchain(&mut surface, swap_config);

        // Setup renderpass and pipeline
        // let set_layout = device.create_descriptor_set_layout(&[
        //         pso::DescriptorSetLayoutBinding {
        //             binding: 0,
        //             ty: pso::DescriptorType::SampledImage,
        //             count: 1,
        //             stage_flags: ShaderStageFlags::FRAGMENT,
        //         },
        //         pso::DescriptorSetLayoutBinding {
        //             binding: 1,
        //             ty: pso::DescriptorType::Sampler,
        //             count: 1,
        //             stage_flags: ShaderStageFlags::FRAGMENT,
        //         },
        //     ],
        let set_layout = device.create_descriptor_set_layout(&[
            pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: pso::DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
            },
        ]);
        //let set_layout = device.create_descriptor_set_layout(&[]);

        let pipeline_layout = device
            .create_pipeline_layout(Some(&set_layout), &[(pso::ShaderStageFlags::VERTEX, 0..8)]);

        let render_pass = {
            let attachment = pass::Attachment {
                format: Some(surface_format),
                ops: pass::AttachmentOps::new(
                    pass::AttachmentLoadOp::Clear,
                    pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: pass::AttachmentOps::DONT_CARE,
                layouts: i::ImageLayout::Undefined..i::ImageLayout::Present,
            };

            let subpass = pass::SubpassDesc {
                colors: &[(0, i::ImageLayout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                preserves: &[],
            };

            let dependency = pass::SubpassDependency {
                passes: pass::SubpassRef::External..pass::SubpassRef::Pass(0),
                stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT
                    ..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                accesses: i::Access::empty()
                    ..(i::Access::COLOR_ATTACHMENT_READ | i::Access::COLOR_ATTACHMENT_WRITE),
            };

            device.create_render_pass(&[attachment], &[subpass], &[dependency])
        };

        //

        // // Descriptors
        let mut desc_pool = device.create_descriptor_pool(
            1, // sets
            &[
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::UniformBuffer,
                    count: 1,
                },
            ],
        );
        let desc_set = desc_pool.allocate_set(&set_layout);

        // Framebuffer and render target creation
        let (frame_images, framebuffers) = match backbuffer {
            Backbuffer::Images(images) => {
                let extent = d::Extent {
                    width: pixel_width as _,
                    height: pixel_height as _,
                    depth: 1,
                };
                let pairs = images
                    .into_iter()
                    .map(|image| {
                        let rtv = device
                            .create_image_view(
                                &image,
                                surface_format,
                                Swizzle::NO,
                                COLOR_RANGE.clone(),
                            )
                            .unwrap();
                        (image, rtv)
                    })
                    .collect::<Vec<_>>();
                let fbos = pairs
                    .iter()
                    .map(|&(_, ref rtv)| {
                        device
                            .create_framebuffer(&render_pass, Some(rtv), extent)
                            .unwrap()
                    })
                    .collect();
                (pairs, fbos)
            }
            Backbuffer::Framebuffer(fbo) => (Vec::new(), vec![fbo]),
        };

        // Buffer allocations
        println!("Memory types: {:?}", memory_types);

        let buffer_stride = std::mem::size_of::<Vertex>() as u64;
        let buffer_len = QUAD.len() as u64 * buffer_stride;

        let buffer_unbound = device
            .create_buffer(buffer_len, buffer::Usage::VERTEX)
            .unwrap();
        let buffer_req = device.get_buffer_requirements(&buffer_unbound);

        let upload_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(m::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        let buffer_memory = device
            .allocate_memory(upload_type, buffer_req.size)
            .unwrap();
        let vertex_buffer = device
            .bind_buffer_memory(&buffer_memory, 0, buffer_unbound)
            .unwrap();

        // TODO: check transitions: read/write mapping and vertex buffer read
        {
            let mut vertices = device
                .acquire_mapping_writer::<Vertex>(&buffer_memory, 0..buffer_len)
                .unwrap();
            vertices.copy_from_slice(&QUAD);
            device.release_mapping_writer(vertices);
        }

        // Image
        // let img_data = include_bytes!("data/logo.png");

        // let img = image::load(Cursor::new(&img_data[..]), image::PNG).unwrap().to_rgba();
        // let (width, height) = img.dimensions();
        // let kind = i::Kind::D2(width as i::Size, height as i::Size, i::AaMode::Single);
        // let row_alignment_mask = limits.min_buffer_copy_pitch_alignment as u32 - 1;
        // let image_stride = 4usize;
        // let row_pitch = (width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask;
        // let upload_size = (height * row_pitch) as u64;

        // let image_buffer_unbound = device.create_buffer(upload_size, buffer::Usage::TRANSFER_SRC).unwrap();
        // let image_mem_reqs = device.get_buffer_requirements(&image_buffer_unbound);
        // let image_upload_memory = device.allocate_memory(upload_type, image_mem_reqs.size).unwrap();
        // let image_upload_buffer = device.bind_buffer_memory(&image_upload_memory, 0, image_buffer_unbound).unwrap();
        //
        let color_size = std::mem::size_of::<[f32; 4]>() as u64;
        let color_buffer_unbound = device
            .create_buffer(color_size, buffer::Usage::UNIFORM)
            .unwrap();
        let color_mem_reqs = device.get_buffer_requirements(&color_buffer_unbound);
        let color_upload_memory = device
            .allocate_memory(upload_type, color_mem_reqs.size)
            .unwrap();
        let color_upload_buffer = device
            .bind_buffer_memory(&color_upload_memory, 0, color_buffer_unbound)
            .unwrap();

        {
            let mut data = device
                .acquire_mapping_writer::<[f32; 4]>(&color_upload_memory, 0..color_size)
                .unwrap();
            data[0] = [1.0, 1.0, 1.0, 1.0];
            device.release_mapping_writer(data);
        }

        device.update_descriptor_sets::<_, Range<_>>(&[
            pso::DescriptorSetWrite {
                set: &desc_set,
                binding: 0,
                array_offset: 0,
                write: pso::DescriptorWrite::UniformBuffer(vec![(&color_upload_buffer, 0..1)]),
            },
        ]);

        // Rendering setup
        let viewport = command::Viewport {
            rect: command::Rect {
                x: 0,
                y: 0,
                w: pixel_width,
                h: pixel_height,
            },
            depth: 0.0..1.0,
        };

        Quad {
            device,
            render_pass,
            pipeline_layout,
            window,
            viewport,
            events_loop,
            command_pool,
            vertex_buffer,
            desc_set,
            swap_chain,
            framebuffers,
            queue,
            desc_pool,
            instance: _instance,
        }
    }
    pub fn render(&mut self, opt: &Opt) {
        let path = opt.get_shader_path();
        let vertex_path = path.join("vertex").with_extension("spv");
        let fragment_path = path.join(&opt.file_name).with_extension("spv");
        let mut frame_fence = self.device.create_fence(false); // TODO: remove
        let mut frame_semaphore = self.device.create_semaphore();
        let (vertex_entry, fragment_entry) = match opt.compiler {
            ShaderCompiler::Rlsl => ("vertex", "fragment"),
            ShaderCompiler::Glsl => ("main", "main"),
        };
        let pipelines = vec![
            create_graphics_pipeline::<back::Backend>(
                &self.device,
                &self.pipeline_layout,
                &self.render_pass,
                &vertex_path,
                &fragment_path,
                vertex_entry,
                fragment_entry,
            ),
        ];
        let query_pool = self.device
            .create_query_pool(hal::query::QueryType::Timestamp, 1);
        for pipeline in &pipelines {
            let mut running = true;
            while running {
                self.events_loop.poll_events(|event| {
                    if let winit::Event::WindowEvent { event, .. } = event {
                        match event {
                            winit::WindowEvent::KeyboardInput {
                                input:
                                    winit::KeyboardInput {
                                        virtual_keycode: Some(winit::VirtualKeyCode::Escape),
                                        state: winit::ElementState::Pressed,
                                        ..
                                    },
                                ..
                            }
                            | winit::WindowEvent::Closed => running = false,
                            _ => (),
                        }
                    }
                });

                self.device.reset_fence(&frame_fence);
                self.command_pool.reset();
                let frame = self.swap_chain
                    .acquire_frame(FrameSync::Semaphore(&mut frame_semaphore));

                // Rendering
                let submit = {
                    let query = hal::query::Query {
                        pool: &query_pool,
                        id: 0,
                    };
                    let mut cmd_buffer = self.command_pool.acquire_command_buffer(false);
                    cmd_buffer.begin_query(query, hal::query::QueryControl::PRECISE);

                    cmd_buffer.set_viewports(&[self.viewport.clone()]);
                    cmd_buffer.set_scissors(&[self.viewport.rect]);
                    cmd_buffer.bind_graphics_pipeline(&pipeline);
                    cmd_buffer
                        .bind_vertex_buffers(pso::VertexBufferSet(vec![(&self.vertex_buffer, 0)]));
                    cmd_buffer.bind_graphics_descriptor_sets(
                        &self.pipeline_layout,
                        0,
                        Some(&self.desc_set),
                    ); //TODO

                    {
                        let mut encoder = cmd_buffer.begin_renderpass_inline(
                            &self.render_pass,
                            &self.framebuffers[frame.id()],
                            self.viewport.rect,
                            &[
                                command::ClearValue::Color(command::ClearColor::Float([
                                    0.8, 0.8, 0.8, 1.0
                                ])),
                            ],
                        );
                        encoder.draw(0..6, 0..1);
                    }

                    cmd_buffer.finish()
                };

                let submission = Submission::new()
                    .wait_on(&[(&frame_semaphore, PipelineStage::BOTTOM_OF_PIPE)])
                    .submit(Some(submit));
                self.queue.submit(submission, Some(&mut frame_fence));

                // TODO: replace with semaphore
                self.device.wait_for_fence(&frame_fence, !0);

                // present frame
                self.swap_chain.present(&mut self.queue, &[]);
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ShaderCompiler {
    Rlsl,
    Glsl,
}
use std::path::PathBuf;
use structopt::StructOpt;
use std::str::FromStr;
use std::string::ParseError;
use std::fmt::Display;
#[derive(Debug, Copy, Clone)]
pub struct ParseErrorShaderCompiler;
impl FromStr for ShaderCompiler {
    type Err = ParseErrorShaderCompiler;
    fn from_str(s: &str) -> Result<ShaderCompiler, Self::Err> {
        match s {
            "rlsl" => Ok(ShaderCompiler::Rlsl),
            "glsl" => Ok(ShaderCompiler::Glsl),
            _ => Err(ParseErrorShaderCompiler),
        }
    }
}

use std::fmt::{Debug, Error, Formatter};
impl Display for ParseErrorShaderCompiler {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        Debug::fmt(self, f)
    }
}

/// A basic example
#[derive(StructOpt, Debug)]
#[structopt(name = "basic")]
pub struct Opt {
    file_name: String,
    #[structopt(short = "c", long = "compiler", default_value = "rlsl", parse(try_from_str))]
    compiler: ShaderCompiler,
}

impl Opt {
    pub fn get_shader_path(&self) -> PathBuf {
        match self.compiler {
            ShaderCompiler::Rlsl => Path::new("../../issues/.shaders/"),
            ShaderCompiler::Glsl => Path::new("../../issues/.shaders-glsl/"),
        }.into()
    }
}
fn main() {
    let opt = Opt::from_args();
    let path = opt.get_shader_path();
    let mut quad = Quad::new();
    quad.render(&opt);
}
