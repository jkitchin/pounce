//! GPU batched dense Cholesky+solve in f32 (wgpu / WGSL). Compiled only
//! under `--features gpu`. One invocation per batch element factors its
//! n×n SPD matrix in place and solves its RHS — an intentionally simple
//! spike kernel (global-memory, no tiling), so its throughput is a
//! *lower bound* on what a tuned kernel would reach.

use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;

const WGSL: &str = r#"
struct Params { n: u32, b: u32, _pad0: u32, _pad1: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> mats: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhs:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x;
  if (e >= params.b) { return; }
  let n = params.n;
  let mbase = e * n * n;
  let vbase = e * n;

  // Cholesky factor (lower triangle) in place.
  for (var j: u32 = 0u; j < n; j = j + 1u) {
    var diag = mats[mbase + j*n + j];
    for (var p: u32 = 0u; p < j; p = p + 1u) {
      let l = mats[mbase + j*n + p];
      diag = diag - l*l;
    }
    let ljj = sqrt(diag);
    mats[mbase + j*n + j] = ljj;
    for (var i: u32 = j + 1u; i < n; i = i + 1u) {
      var s = mats[mbase + i*n + j];
      for (var p: u32 = 0u; p < j; p = p + 1u) {
        s = s - mats[mbase + i*n + p] * mats[mbase + j*n + p];
      }
      mats[mbase + i*n + j] = s / ljj;
    }
  }
  // forward L y = b
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    var s = rhs[vbase + i];
    for (var p: u32 = 0u; p < i; p = p + 1u) {
      s = s - mats[mbase + i*n + p] * rhs[vbase + p];
    }
    rhs[vbase + i] = s / mats[mbase + i*n + i];
  }
  // back Lᵀ x = y
  for (var ii: u32 = 0u; ii < n; ii = ii + 1u) {
    let i = n - 1u - ii;
    var s = rhs[vbase + i];
    for (var p: u32 = i + 1u; p < n; p = p + 1u) {
      s = s - mats[mbase + p*n + i] * rhs[vbase + p];
    }
    rhs[vbase + i] = s / mats[mbase + i*n + i];
  }
}
"#;

pub struct GpuContext {
    pub backend: String,
    pub adapter_name: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

fn parse_backends(pref: &str) -> wgpu::Backends {
    match pref.to_ascii_lowercase().as_str() {
        "vulkan" | "vk" => wgpu::Backends::VULKAN,
        "metal" | "mtl" => wgpu::Backends::METAL,
        "dx12" | "d3d12" => wgpu::Backends::DX12,
        "gl" | "opengl" => wgpu::Backends::GL,
        "primary" => wgpu::Backends::PRIMARY,
        _ => wgpu::Backends::all(),
    }
}

impl GpuContext {
    /// Probe-and-verify init: enumerate an adapter on the requested
    /// backend(s), build the pipeline, and run a tiny known system. Any
    /// failure (no adapter, device error, wrong answer) returns `None`
    /// so the caller falls back to CPU — the §11 runtime-selection
    /// contract, exercised.
    pub fn new(backend_pref: &str) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: parse_backends(backend_pref),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gpu_spike"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .ok()?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("batched_chol"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        let ctx = GpuContext {
            backend: format!("{:?}", info.backend),
            adapter_name: info.name,
            device,
            queue,
            pipeline,
            bgl,
        };

        // Self-check: solve [[2,1],[1,3]] x = [1,1] -> x ≈ [0.4, 0.2].
        let mut mat = vec![2.0f32, 1.0, 1.0, 3.0];
        let mut rhs = vec![1.0f32, 1.0];
        ctx.solve(2, 1, &mut mat, &mut rhs).ok()?;
        if (rhs[0] - 0.4).abs() > 1e-3 || (rhs[1] - 0.2).abs() > 1e-3 {
            return None;
        }
        Some(ctx)
    }

    /// Factor+solve `b` SPD systems of size `n`. `mats` is `b*n*n`
    /// row-major f32 (overwritten with the factor); `rhs` is `b*n`
    /// (overwritten with the solution). Returns the on-device wall time
    /// including upload + dispatch + readback (the honest end-to-end
    /// cost the crossover must beat — and where unified memory helps).
    pub fn solve(
        &self,
        n: usize,
        b: usize,
        mats: &mut [f32],
        rhs: &mut [f32],
    ) -> Result<Duration, String> {
        let t0 = Instant::now();
        let params = [n as u32, b as u32, 0u32, 0u32];
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mats_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mats"),
                contents: bytemuck::cast_slice(mats),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let rhs_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rhs"),
                contents: bytemuck::cast_slice(rhs),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (rhs.len() * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &self.bgl,
            entries: &[
                bind(0, &params_buf),
                bind(1, &mats_buf),
                bind(2, &rhs_buf),
            ],
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = ((b as u32) + 63) / 64;
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        enc.copy_buffer_to_buffer(&rhs_buf, 0, &readback, 0, (rhs.len() * 4) as u64);
        self.queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| format!("{:?}", e))?;
        let data = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&data);
        rhs.copy_from_slice(out);
        drop(data);
        readback.unmap();
        Ok(t0.elapsed())
    }
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}
