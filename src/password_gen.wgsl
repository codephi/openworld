// Shader WGSL para gerar senhas na GPU

struct Params {
    charset_len: u32,
    password_len: u32,
    start_index: u32,
    batch_size: u32,
}

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> charset: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn generate_passwords(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.batch_size) {
        return;
    }
    
    let password_index = params.start_index + idx;
    var temp_index = password_index;
    
    // Escreve a senha diretamente no buffer de saída
    let output_offset = idx * params.password_len;
    
    // Gera a senha de trás para frente
    for (var i: u32 = 0u; i < params.password_len; i = i + 1u) {
        let char_index = temp_index % params.charset_len;
        let pos = params.password_len - 1u - i;
        output[output_offset + pos] = charset[char_index];
        temp_index = temp_index / params.charset_len;
    }
}
