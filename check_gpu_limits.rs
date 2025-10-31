use pollster;
use wgpu;

fn main() {
    let instance = wgpu::Instance::default();
    
    let adapter = pollster::block_on(async {
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
    });
    
    if let Some(adapter) = adapter {
        let info = adapter.get_info();
        println!("=== GPU Info ===");
        println!("Name: {}", info.name);
        println!("Vendor: {:?}", info.vendor);
        println!("Device: {:?}", info.device);
        println!("Device Type: {:?}", info.device_type);
        println!("Driver: {}", info.driver);
        println!("Driver Info: {}", info.driver_info);
        println!("Backend: {:?}", info.backend);
        
        let limits = adapter.limits();
        println!("\n=== Compute Limits ===");
        println!("Max workgroup size X: {}", limits.max_compute_workgroup_size_x);
        println!("Max workgroup size Y: {}", limits.max_compute_workgroup_size_y);
        println!("Max workgroup size Z: {}", limits.max_compute_workgroup_size_z);
        println!("Max workgroups per dimension: {}", limits.max_compute_workgroups_per_dimension);
        println!("Max invocations per workgroup: {}", limits.max_compute_invocations_per_workgroup);
        
        println!("\n=== Memory Limits ===");
        println!("Max buffer size: {} GB", limits.max_buffer_size / (1024 * 1024 * 1024));
        println!("Max storage buffer binding size: {} MB", limits.max_storage_buffer_binding_size / (1024 * 1024));
        println!("Max uniform buffer binding size: {} KB", limits.max_uniform_buffer_binding_size / 1024);
        println!("Max push constant size: {} bytes", limits.max_push_constant_size);
        
        println!("\n=== Other Limits ===");
        println!("Max bind groups: {}", limits.max_bind_groups);
        println!("Max bindings per bind group: {}", limits.max_bindings_per_bind_group);
        println!("Max dynamic storage buffers per pipeline: {}", limits.max_dynamic_storage_buffers_per_pipeline_layout);
        println!("Max storage buffers per shader stage: {}", limits.max_storage_buffers_per_shader_stage);
        
        let features = adapter.features();
        println!("\n=== Supported Features ===");
        println!("Features: {:?}", features);
    } else {
        println!("No GPU adapter found!");
    }
}