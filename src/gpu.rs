// Módulo para processamento em GPU real usando wgpu

#[cfg(feature = "gpu")]
use bytemuck::{NoUninit, Pod, Zeroable};
use rayon::prelude::*;
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

pub struct GpuInfo {
    pub available: bool,
    pub name: String,
    pub memory: u64,
    pub compute_units: u32,
}

// Detecta GPU usando WMI no Windows
pub fn detect_gpu() -> GpuInfo {
    use std::process::Command;

    // Tenta obter informações da GPU via WMI
    let output = Command::new("wmic")
        .args(&["path", "win32_VideoController", "get", "name,AdapterRAM"])
        .output();

    match output {
        Ok(result) => {
            let info = String::from_utf8_lossy(&result.stdout);
            let lines: Vec<&str> = info.lines().collect();

            for line in lines.iter().skip(1) {
                let trimmed = line.trim();
                if !trimmed.is_empty() && trimmed != "Name" {
                    // Procura por GPUs conhecidas
                    if trimmed.contains("NVIDIA")
                        || trimmed.contains("AMD")
                        || trimmed.contains("Radeon")
                        || trimmed.contains("GeForce")
                    {
                        // Extrai o nome
                        let parts: Vec<&str> = trimmed.split_whitespace().collect();
                        let name = parts[0..parts.len().saturating_sub(1)].join(" ");

                        // Estima memória (WMI retorna em bytes, mas nem sempre é preciso)
                        let memory = if trimmed.contains("4294967296") {
                            4 * 1024 * 1024 * 1024u64 // 4GB
                        } else if trimmed.contains("8589934592") {
                            8 * 1024 * 1024 * 1024u64 // 8GB
                        } else {
                            2 * 1024 * 1024 * 1024u64 // Default 2GB
                        };

                        return GpuInfo {
                            available: true,
                            name: name.to_string(),
                            memory,
                            compute_units: 16, // Valor estimado
                        };
                    }
                }
            }

            // Se encontrou alguma placa de vídeo mas não GPU dedicada
            if info.contains("Graphics") || info.contains("Display") {
                return GpuInfo {
                    available: true,
                    name: "Integrated Graphics".to_string(),
                    memory: 512 * 1024 * 1024, // 512MB estimado
                    compute_units: 8,
                };
            }
        }
        Err(_) => {}
    }

    GpuInfo {
        available: false,
        name: "No GPU detected".to_string(),
        memory: 0,
        compute_units: 0,
    }
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    charset_len: u32,
    password_len: u32,
    start_index: u32,
    batch_size: u32,
}

// Gerador de senhas GPU real com wgpu
#[cfg(feature = "gpu")]
pub struct GpuPasswordGenerator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    charset: String,
    charset_len: usize,
    workgroup_size: u32,
}

// Fallback para quando GPU não está disponível
#[cfg(not(feature = "gpu"))]
pub struct GpuPasswordGenerator {
    charset: String,
    charset_len: usize,
}

#[cfg(feature = "gpu")]
impl GpuPasswordGenerator {
    pub fn new(charset: &str, mut workgroup_size: u32) -> Result<Self, String> {
        // Inicializa wgpu primeiro para verificar limites reais
        let instance = wgpu::Instance::default();
        
        let adapter = pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
        })
        .ok_or("Failed to find GPU adapter".to_string())?;
        
        // Obtém limites reais do dispositivo
        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("GPU Password Generator"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                    },
                    None,
                )
                .await
        })
        .map_err(|e| format!("Failed to create device: {}", e))?;
        
        // Verifica limites reais do dispositivo
        let limits = device.limits();
        let max_workgroup_size = limits.max_compute_invocations_per_workgroup
            .min(limits.max_compute_workgroup_size_x);
        
        // Ajusta workgroup_size se necessário
        if workgroup_size > max_workgroup_size {
            eprintln!("⚠️  Workgroup size {} excede o limite da GPU ({}). Ajustando...", 
                workgroup_size, max_workgroup_size);
            // Encontra a maior potência de 2 que cabe no limite
            workgroup_size = 1;
            while workgroup_size * 2 <= max_workgroup_size {
                workgroup_size *= 2;
            }
            eprintln!("   Usando workgroup size: {}", workgroup_size);
        }
        
        // Valida workgroup_size
        if !workgroup_size.is_power_of_two() || workgroup_size == 0 {
            return Err(format!("Workgroup size deve ser potência de 2. Recebido: {}", workgroup_size));
        }
        
        Ok(Self {
            device,
            queue,
            charset: charset.to_string(),
            charset_len: charset.len(),
            workgroup_size,
        })
    }

    fn create_shader(&self) -> wgpu::ShaderModule {
        // Shader com workgroup_size dinâmico
        let shader_source = format!(
            r#"
struct Params {{
    charset_len: u32,
    password_len: u32,
    start_index: u32,
    batch_size: u32,
}}

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> charset: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size({})
fn generate_passwords(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    
    if (idx >= params.batch_size) {{
        return;
    }}
    
    let password_index = params.start_index + idx;
    var temp_index = password_index;
    
    let output_offset = idx * params.password_len;
    
    for (var i: u32 = 0u; i < params.password_len; i = i + 1u) {{
        let char_index = temp_index % params.charset_len;
        let pos = params.password_len - 1u - i;
        output[output_offset + pos] = charset[char_index];
        temp_index = temp_index / params.charset_len;
    }}
}}
"#,
            self.workgroup_size
        );
        
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Password Generator Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        })
    }
    
    pub fn generate_batch(
        &self,
        password_len: usize,
        start_index: usize,
        batch_size: usize,
    ) -> Result<Vec<String>, String> {
        // Prepara dados do charset
        let charset_data: Vec<u32> = self.charset.chars().map(|c| c as u32).collect();

        // Cria buffers GPU
        let params = GpuParams {
            charset_len: self.charset_len as u32,
            password_len: password_len as u32,
            start_index: start_index as u32,
            batch_size: batch_size as u32,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let charset_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Charset Buffer"),
                contents: bytemuck::cast_slice(&charset_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = batch_size * password_len * 4; // 4 bytes por caractere
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Cria pipeline
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Password Gen Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Password Gen Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: charset_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Cria shader dinamicamente com o workgroup_size correto
        let shader = self.create_shader();
        
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Password Gen Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Password Gen Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "generate_passwords",
            });

        // Executa shader
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Password Gen Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Password Gen Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // Usa workgroup_size configurado para calcular número de workgroups
            let workgroups = (batch_size as u32 + self.workgroup_size - 1) / self.workgroup_size;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Cria buffer para leitura
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size as u64);

        self.queue.submit(Some(encoder.finish()));

        // Lê resultados
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .unwrap()
            .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        // Converte resultados em strings
        let mut passwords = Vec::new();
        for i in 0..batch_size {
            let start = i * password_len;
            let end = start + password_len;
            let password: String = result[start..end]
                .iter()
                .map(|&c| char::from_u32(c).unwrap_or('?'))
                .collect();
            passwords.push(password);
        }

        Ok(passwords)
    }
}

#[cfg(not(feature = "gpu"))]
impl GpuPasswordGenerator {
    pub fn new(charset: &str) -> Result<Self, String> {
        Ok(Self {
            charset: charset.to_string(),
            charset_len: charset.len(),
        })
    }

    // Simula geração em batch como se fosse GPU
    pub fn generate_batch(
        &self,
        password_len: usize,
        start_index: usize,
        batch_size: usize,
    ) -> Result<Vec<String>, String> {
        let chars: Vec<char> = self.charset.chars().collect();

        // Gera senhas em paralelo simulando GPU
        let passwords: Vec<String> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let mut index = start_index + i;
                let mut password = String::with_capacity(password_len);

                for _ in 0..password_len {
                    password.push(chars[index % self.charset_len]);
                    index /= self.charset_len;
                }

                password.chars().rev().collect()
            })
            .collect();

        Ok(passwords)
    }

    // Simula processamento paralelo massivo
    pub fn generate_parallel(
        &self,
        password_len: usize,
        total_passwords: usize,
        batch_size: usize,
    ) -> Vec<String> {
        let mut all_passwords = Vec::with_capacity(total_passwords);
        let num_batches = (total_passwords + batch_size - 1) / batch_size;

        for batch in 0..num_batches {
            let start = batch * batch_size;
            let size = batch_size.min(total_passwords - start);

            if let Ok(mut batch_passwords) = self.generate_batch(password_len, start, size) {
                all_passwords.append(&mut batch_passwords);
            }
        }

        all_passwords
    }
}

// Versão CPU fallback quando GPU não está disponível
pub fn generate_passwords_cpu(
    charset: &str,
    length: usize,
    start_index: usize,
    count: usize,
) -> Vec<String> {
    let chars: Vec<char> = charset.chars().collect();
    let base = chars.len();
    let mut passwords = Vec::with_capacity(count);

    for i in 0..count {
        let mut index = start_index + i;
        let mut password = String::with_capacity(length);

        for _ in 0..length {
            password.push(chars[index % base]);
            index /= base;
        }

        passwords.push(password.chars().rev().collect());
    }

    passwords
}
