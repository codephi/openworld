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
        
        println!("\n=== Recomendações ===");
        println!("Workgroup size máximo recomendado: {} (deve ser potência de 2)", 
            limits.max_compute_invocations_per_workgroup.min(limits.max_compute_workgroup_size_x));
        println!("Batch size máximo teórico: {} MB de memória",
            limits.max_storage_buffer_binding_size / (1024 * 1024));
    } else {
        println!("No GPU adapter found!");
    }
}