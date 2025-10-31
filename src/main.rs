mod gpu;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::collections::HashMap;

static FOUND: AtomicBool = AtomicBool::new(false);
static TESTED: AtomicUsize = AtomicUsize::new(0);

#[derive(Serialize, Deserialize, Debug, Default)]
struct Metrics {
    start_time: String,
    end_time: Option<String>,
    total_duration_seconds: f64,
    phases: HashMap<String, PhaseMetrics>,
    total_passwords_generated: usize,
    total_passwords_tested: usize,
    passwords_per_second: f64,
    password_found: Option<String>,
    charset_used: String,
    min_length: usize,
    max_length: usize,
    gpu_used: bool,
    threads_used: usize,
    partitions: usize,
    partition_id: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct PhaseMetrics {
    name: String,
    start_time: String,
    end_time: String,
    duration_seconds: f64,
    items_processed: Option<usize>,
    items_per_second: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Checkpoint {
    file_path: String,
    charset: String,
    min_length: usize,
    max_length: usize,
    current_length: usize,
    last_tested: Option<String>,
    total_tested: usize,
    total_combinations: usize,
    start_time: String,
    last_update: String,
}

#[derive(Parser, Debug)]
#[command(name = "brute_force")]
#[command(author = "Universal Brute Force Tool")]
#[command(version = "1.0")]
#[command(about = "Ferramenta universal de força bruta com suporte a GPU", long_about = None)]
struct Args {
    /// Arquivo com senhas para testar (depreciado - use --command)
    #[arg(short, long)]
    file: Option<PathBuf>,
    
    /// Comando customizado para executar (use $password para a senha)
    #[arg(long)]
    command: Option<String>,

    /// Tamanho mínimo da senha
    #[arg(short = 'm', long, default_value_t = 1)]
    min: usize,

    /// Tamanho máximo da senha
    #[arg(short = 'x', long, default_value_t = 8)]
    max: usize,

    /// Incluir letras minúsculas (a-z)
    #[arg(long = "az")]
    lowercase: bool,

    /// Incluir letras maiúsculas (A-Z)
    #[arg(long = "AZ")]
    uppercase: bool,

    /// Incluir números (0-9)
    #[arg(long = "09")]
    numbers: bool,

    /// Incluir caracteres especiais (!@#$%^&*()_+-=[]{}|;:'",.<>?/`~)
    #[arg(long)]
    specials: bool,

    /// Caracteres customizados para incluir
    #[arg(long)]
    custom: Option<String>,

    /// Lista de senhas para testar (arquivo texto, uma por linha)
    #[arg(short = 'w', long)]
    wordlist: Option<PathBuf>,

    /// Número de threads (padrão: todos os cores)
    #[arg(short = 't', long)]
    threads: Option<usize>,

    /// Número máximo de CPUs para usar (padrão: todos)
    #[arg(long = "max-cpus")]
    max_cpus: Option<usize>,

    /// Usar GPU para geração de combinações (se disponível)
    #[arg(long)]
    gpu: bool,

    /// Tamanho do workgroup na GPU (threads por grupo)
    #[arg(long, default_value_t = 64)]
    workgroup_size: u32,

    /// Tamanho do batch para processar na GPU por vez
    #[arg(long, default_value_t = 65536)]
    gpu_batch_size: usize,

    /// Número de partições paralelas para dividir o espaço de busca
    #[arg(long, default_value_t = 1)]
    parallel: usize,

    /// ID da partição atual (0 a parallel-1)
    #[arg(long)]
    partition_id: Option<usize>,

    /// Modo streaming (não armazena todas as senhas em memória)
    #[arg(long)]
    stream: bool,

    /// Salvar combinações em arquivo temporário (para grandes conjuntos)
    #[arg(long)]
    disk_cache: bool,

    /// Salvar dicionário gerado em arquivo
    #[arg(long)]
    save_dictionary: Option<PathBuf>,

    /// Apenas gerar dicionário sem testar senhas
    #[arg(long)]
    generate_only: bool,

    /// Salvar progresso para poder retomar depois
    #[arg(long)]
    save_progress: bool,

    /// Retomar de um checkpoint anterior
    #[arg(long)]
    resume: bool,

    /// Arquivo de checkpoint (padrão: rar_cracker_checkpoint.json)
    #[arg(long, default_value = "rar_cracker_checkpoint.json")]
    checkpoint_file: PathBuf,

    /// Modo verboso
    #[arg(short, long)]
    verbose: bool,
}

// Função legada - mantida para compatibilidade
fn test_password(_target: &str, _password: &str) -> bool {
    // Esta função foi depreciada em favor de test_password_with_command
    // Use --command para especificar o comando a executar
    eprintln!("⚠️  Função test_password depreciada. Use --command.");
    false
}

fn test_password_with_command(command: &str, password: &str, verbose: bool) -> bool {
    if FOUND.load(Ordering::Relaxed) {
        return false;
    }
    
    // Substitui $password pela senha atual
    let cmd = command.replace("$password", password);
    
    TESTED.fetch_add(1, Ordering::Relaxed);
    
    if verbose {
        eprintln!("  🔧 Executando: {}", cmd);
    }
    
    // Separa comando e argumentos
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return false;
    }
    
    let output = if cfg!(target_os = "windows") {
        // Windows: usa cmd /c para executar
        Command::new("cmd")
            .args(&["/c", &cmd])
            .output()
    } else {
        // Unix: usa sh -c
        Command::new("sh")
            .args(&["-c", &cmd])
            .output()
    };
    
    match output {
        Ok(result) => {
            if result.status.success() {
                FOUND.store(true, Ordering::Relaxed);
                true
            } else {
                false
            }
        }
        Err(e) => {
            if verbose {
                eprintln!("    ⚠️  Erro ao executar comando: {}", e);
            }
            false
        }
    }
}

fn build_charset(args: &Args) -> String {
    let mut charset = String::new();

    if args.lowercase {
        charset.push_str("abcdefghijklmnopqrstuvwxyz");
    }
    if args.uppercase {
        charset.push_str("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    }
    if args.numbers {
        charset.push_str("0123456789");
    }
    if args.specials {
        charset.push_str("!@#$%^&*()_+-=[]{}|;:'\",.<>?/`~");
    }
    if let Some(custom) = &args.custom {
        charset.push_str(custom);
    }

    // Remove duplicados
    let unique_chars: String = charset.chars().unique().collect();
    unique_chars
}

fn generate_passwords_bruteforce(charset: &str, min_len: usize, max_len: usize) -> Vec<String> {
    generate_passwords_bruteforce_with_gpu(charset, min_len, max_len, false, 64, 65536)
}

fn generate_passwords_bruteforce_with_gpu(
    charset: &str, 
    min_len: usize, 
    max_len: usize, 
    use_gpu: bool,
    workgroup_size: u32,
    batch_size: usize,
) -> Vec<String> {
    let mut passwords = Vec::new();
    let chars: Vec<char> = charset.chars().collect();

    println!("📊 Charset: {}", charset);
    println!("📏 Comprimento: {} a {} caracteres", min_len, max_len);

    // Calcula o total de combinações
    let mut total: usize = 0;
    for len in min_len..=max_len {
        let combinations = chars.len().pow(len as u32);
        total = total.saturating_add(combinations);
    }

    println!("🔢 Total de combinações: {}", format_number(total));

    if total > 100_000_000 {
        println!("⚠️  AVISO: Mais de 100 milhões de combinações!");
        println!("⚠️  Isso pode levar MUITO tempo. Considere reduzir o escopo.");
        println!("\nDeseja continuar? (s/n): ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        if !input.trim().eq_ignore_ascii_case("s") {
            return vec![];
        }
    }

    // Tentar usar GPU se disponível
    if use_gpu {
        println!("\n🎮 Tentando usar GPU para geração...");
        
        match gpu::GpuPasswordGenerator::new(charset, workgroup_size) {
            Ok(gpu_gen) => {
                println!("✅ GPU inicializada com sucesso!");
                println!("⚙️  Workgroup size: {} threads", workgroup_size);
                println!("📦 Batch size: {} senhas por vez", format_number(batch_size));
                println!("🚀 Gerando senhas na GPU...");
                
                let pb = ProgressBar::new(total as u64);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) GPU: {msg}")
                        .unwrap()
                        .progress_chars("#>-"),
                );
                
                for len in min_len..=max_len {
                    let len_total = chars.len().pow(len as u32);
                    pb.set_message(format!("Gerando senhas de {} caracteres", len));
                    
                    let mut generated = 0;
                    while generated < len_total {
                        let current_batch = batch_size.min(len_total - generated);
                        
                        match gpu_gen.generate_batch(len, generated, current_batch) {
                            Ok(mut batch) => {
                                passwords.append(&mut batch);
                                pb.inc(current_batch as u64);
                                generated += current_batch;
                            }
                            Err(e) => {
                                println!("\n⚠️ Erro na GPU: {}. Voltando para CPU...", e);
                                return generate_passwords_bruteforce_cpu_fallback(charset, min_len, max_len);
                            }
                        }
                    }
                }
                
                pb.finish_with_message(format!("✅ {} senhas geradas na GPU!", format_number(passwords.len())));
                println!();
                return passwords;
            }
            Err(e) => {
                println!("⚠️ GPU não disponível: {}. Usando CPU...", e);
            }
        }
    }
    
    // Fallback para CPU
    generate_passwords_bruteforce_cpu_fallback(charset, min_len, max_len)
}

fn generate_passwords_bruteforce_cpu_fallback(charset: &str, min_len: usize, max_len: usize) -> Vec<String> {
    let mut passwords = Vec::new();
    let chars: Vec<char> = charset.chars().collect();
    
    let mut total: usize = 0;
    for len in min_len..=max_len {
        total = total.saturating_add(chars.len().pow(len as u32));
    }
    
    println!("\n💻 Gerando combinações na CPU...");
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    for len in min_len..=max_len {
        let _current_total = chars.len().pow(len as u32);
        pb.set_message(format!("Gerando senhas de {} caracteres", len));
        
        for combination in (0..len).map(|_| chars.iter()).multi_cartesian_product() {
            let password: String = combination.into_iter().collect();
            passwords.push(password);
            pb.inc(1);
        }
    }

    pb.finish_with_message(format!("✅ {} combinações geradas", format_number(passwords.len())));
    println!();

    passwords
}

fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push('.');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

// Salva dicionário de senhas em arquivo
fn save_dictionary_to_file(passwords: &[String], filepath: &PathBuf) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::{BufWriter, Write};
    
    println!("💾 Salvando {} senhas em {:?}...", format_number(passwords.len()), filepath);
    
    let file = File::create(filepath)?;
    let mut writer = BufWriter::new(file);
    
    let pb = ProgressBar::new(passwords.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) Salvando...")
            .unwrap()
            .progress_chars("#>-"),
    );
    
    for password in passwords {
        writeln!(writer, "{}", password)?;
        pb.inc(1);
    }
    
    writer.flush()?;
    pb.finish_with_message("✅ Dicionário salvo com sucesso!");
    
    // Mostrar tamanho do arquivo
    if let Ok(metadata) = std::fs::metadata(filepath) {
        let size = metadata.len();
        let size_mb = size as f64 / (1024.0 * 1024.0);
        println!("📁 Tamanho do arquivo: {:.2} MB", size_mb);
    }
    
    Ok(())
}

// Calcula as sementes iniciais para cada partição
fn calculate_partition_seeds(charset: &str, length: usize, num_partitions: usize) -> Vec<String> {
    let chars: Vec<char> = charset.chars().collect();
    let base = chars.len();
    let total_combinations = base.pow(length as u32);
    let partition_size = total_combinations / num_partitions;
    
    let mut seeds = Vec::new();
    
    println!("🌱 Calculando sementes para {} partições:", num_partitions);
    println!("   Base: {} caracteres", base);
    println!("   Comprimento: {} caracteres", length);
    println!("   Total de combinações: {}", format_number(total_combinations));
    println!("   Tamanho por partição: {}\n", format_number(partition_size));
    
    for i in 0..num_partitions {
        let start_index = i * partition_size;
        let seed = index_to_password(start_index, &chars, length);
        seeds.push(seed.clone());
        println!("   Partição {}: {} (index: {})", i, seed, format_number(start_index));
    }
    
    seeds
}

// Converte um índice numérico em uma senha
fn index_to_password(mut index: usize, chars: &[char], length: usize) -> String {
    let base = chars.len();
    let mut password = String::new();
    
    for _ in 0..length {
        password.push(chars[index % base]);
        index /= base;
    }
    
    password.chars().rev().collect()
}

// Converte uma senha em um índice numérico
fn password_to_index(password: &str, chars: &[char]) -> usize {
    let base = chars.len();
    let char_to_index: std::collections::HashMap<char, usize> = 
        chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    
    password.chars().rev().enumerate().fold(0, |acc, (i, c)| {
        acc + char_to_index.get(&c).unwrap_or(&0) * base.pow(i as u32)
    })
}

// Gera senhas para uma partição específica
fn generate_partition_passwords(
    charset: &str, 
    length: usize, 
    partition_id: usize, 
    num_partitions: usize
) -> Vec<String> {
    let chars: Vec<char> = charset.chars().collect();
    let base = chars.len();
    let total_combinations = base.pow(length as u32);
    let partition_size = total_combinations / num_partitions;
    
    let start_index = partition_id * partition_size;
    let end_index = if partition_id == num_partitions - 1 {
        total_combinations
    } else {
        (partition_id + 1) * partition_size
    };
    
    let mut passwords = Vec::new();
    
    for index in start_index..end_index {
        passwords.push(index_to_password(index, &chars, length));
    }
    
    passwords
}

fn save_checkpoint(checkpoint: &Checkpoint, path: &PathBuf) -> std::io::Result<()> {
    let mut checkpoint = checkpoint.clone();
    checkpoint.last_update = chrono::Local::now().to_rfc3339();
    
    let json = serde_json::to_string_pretty(&checkpoint)?;
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

fn load_checkpoint(path: &PathBuf) -> std::io::Result<Checkpoint> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let checkpoint: Checkpoint = serde_json::from_reader(reader)?;
    Ok(checkpoint)
}

// Gera combinações e salva em arquivo
fn generate_passwords_to_disk(charset: &str, min_len: usize, max_len: usize) -> Result<PathBuf, std::io::Error> {
    let chars: Vec<char> = charset.chars().collect();
    let temp_file = std::env::temp_dir().join("rar_cracker_passwords.txt");
    let mut file = File::create(&temp_file)?;
    
    let mut total: usize = 0;
    for len in min_len..=max_len {
        total = total.saturating_add(chars.len().pow(len as u32));
    }
    
    println!("💾 Salvando combinações em: {:?}", temp_file);
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) Salvando em disco")
            .unwrap()
            .progress_chars("#>-"),
    );
    
    for len in min_len..=max_len {
        for combination in (0..len).map(|_| chars.iter()).multi_cartesian_product() {
            let password: String = combination.into_iter().collect();
            writeln!(file, "{}", password)?;
            pb.inc(1);
        }
    }
    
    pb.finish_with_message("✅ Salvo em disco");
    Ok(temp_file)
}

// Modo streaming com suporte a checkpoint e partições
fn bruteforce_streaming_with_checkpoint(
    rar_path: &str,
    charset: &str,
    min_len: usize,
    max_len: usize,
    verbose: bool,
    save_progress: bool,
    checkpoint_file: &PathBuf,
    resume: bool,
    partition_id: Option<usize>,
    num_partitions: usize,
    command: Option<&str>,
) -> Option<String> {
    let chars: Vec<char> = charset.chars().collect();
    
    let mut total: usize = 0;
    for len in min_len..=max_len {
        total = total.saturating_add(chars.len().pow(len as u32));
    }
    
    // Carregar checkpoint se existir
    let (start_length, skip_until) = if resume && checkpoint_file.exists() {
        match load_checkpoint(checkpoint_file) {
            Ok(cp) => {
                println!("🔄 Retomando de checkpoint:");
                println!("   Última testada: {:?}", cp.last_tested);
                println!("   Progresso anterior: {} de {}", format_number(cp.total_tested), format_number(cp.total_combinations));
                println!("   Comprimento atual: {}\n", cp.current_length);
                (cp.current_length, cp.last_tested)
            }
            Err(e) => {
                println!("⚠️  Erro ao carregar checkpoint: {}. Iniciando do zero.\n", e);
                (min_len, None)
            }
        }
    } else {
        (min_len, None)
    };
    
    println!("🎯 Modo streaming: gerando e testando sob demanda");
    println!("💾 Uso de memória: Mínimo (não armazena combinações)");
    if save_progress {
        println!("💾 Checkpoint habilitado: {}", checkpoint_file.display());
    }
    println!();
    
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    
    let checkpoint = Arc::new(Mutex::new(Checkpoint {
        file_path: rar_path.to_string(),
        charset: charset.to_string(),
        min_length: min_len,
        max_length: max_len,
        current_length: start_length,
        last_tested: skip_until.clone(),
        total_tested: 0,
        total_combinations: total,
        start_time: chrono::Local::now().to_rfc3339(),
        last_update: chrono::Local::now().to_rfc3339(),
    }));
    
    let should_skip = Arc::new(AtomicBool::new(skip_until.is_some()));
    let save_interval = Arc::new(AtomicUsize::new(0));
    
    for len in start_length..=max_len {
        // Gerar senhas considerando partições
        let passwords = if num_partitions > 1 {
            if let Some(id) = partition_id {
                generate_partition_passwords(charset, len, id, num_partitions)
            } else {
                // Se não especificou partição, gera todas
                (0..len)
                    .map(|_| chars.iter())
                    .multi_cartesian_product()
                    .map(|combination| combination.into_iter().collect::<String>())
                    .collect()
            }
        } else {
            (0..len)
                .map(|_| chars.iter())
                .multi_cartesian_product()
                .map(|combination| combination.into_iter().collect::<String>())
                .collect()
        };
        
        // Atualizar comprimento atual no checkpoint
        if let Ok(mut cp) = checkpoint.lock() {
            cp.current_length = len;
        }
        
        let checkpoint_clone = Arc::clone(&checkpoint);
        let checkpoint_file_clone = checkpoint_file.clone();
        let skip_clone = Arc::clone(&should_skip);
        let skip_until_clone = skip_until.clone();
        let save_interval_clone = Arc::clone(&save_interval);
        let pb_clone = pb.clone();
        
        let found = passwords.par_iter().find_any(move |password| {
            // Skip até encontrar onde parou
            if skip_clone.load(Ordering::Relaxed) {
                if let Some(ref last) = skip_until_clone {
                    if password.as_str() == last.as_str() {
                        skip_clone.store(false, Ordering::Relaxed);
                    }
                    return false;
                }
            }
            
            pb_clone.inc(1);
            if verbose {
                pb_clone.set_message(format!("Testando: {}", password));
            }
            
            // Salvar checkpoint a cada 10000 senhas testadas
            if save_progress {
                let count = save_interval_clone.fetch_add(1, Ordering::Relaxed);
                if count % 10000 == 0 {
                    if let Ok(mut cp) = checkpoint_clone.lock() {
                        cp.last_tested = Some(password.to_string());
                        cp.total_tested = TESTED.load(Ordering::Relaxed);
                        let _ = save_checkpoint(&cp, &checkpoint_file_clone);
                    }
                }
            }
            
            if let Some(cmd) = command {
                test_password_with_command(cmd, password, verbose)
            } else {
                test_password(rar_path, password)
            }
        });
        
        if found.is_some() {
            pb.finish_with_message("✅ Senha encontrada!");
            // Remover checkpoint em caso de sucesso
            if checkpoint_file.exists() {
                let _ = std::fs::remove_file(checkpoint_file);
            }
            return found.cloned();
        }
    }
    
    pb.finish_with_message("Concluído");
    // Remover checkpoint quando concluir tudo
    if checkpoint_file.exists() {
        let _ = std::fs::remove_file(checkpoint_file);
    }
    None
}

fn check_gpu_availability() -> (bool, String) {
    // Verifica se há GPU disponível usando Windows WMI
    let output = Command::new("wmic")
        .args(&["path", "win32_VideoController", "get", "name"])
        .output();
    
    match output {
        Ok(result) => {
            let gpu_info = String::from_utf8_lossy(&result.stdout);
            let has_gpu = gpu_info.contains("NVIDIA") || 
                         gpu_info.contains("AMD") || 
                         gpu_info.contains("Radeon") || 
                         gpu_info.contains("GeForce");
            
            if has_gpu {
                // Extrai o nome da GPU
                let lines: Vec<&str> = gpu_info.lines().collect();
                for line in lines {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() && trimmed != "Name" {
                        return (true, trimmed.to_string());
                    }
                }
            }
            (false, "Nenhuma GPU compatível detectada".to_string())
        }
        Err(_) => (false, "Não foi possível detectar GPU".to_string()),
    }
}

fn save_metrics(metrics: &Metrics) {
    if let Ok(json) = serde_json::to_string_pretty(metrics) {
        std::fs::write("metadata.json", json).ok();
    }
}

fn update_phase_metrics(
    metrics: &mut Metrics,
    phase_name: &str,
    start: Instant,
    items: Option<usize>,
) {
    let duration = start.elapsed();
    let phase_metrics = PhaseMetrics {
        name: phase_name.to_string(),
        start_time: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        end_time: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        duration_seconds: duration.as_secs_f64(),
        items_processed: items,
        items_per_second: items.map(|i| i as f64 / duration.as_secs_f64()),
    };
    
    metrics.phases.insert(phase_name.to_string(), phase_metrics.clone());
    
    println!("⏱️  {} concluído em {:.2}s", phase_name, duration.as_secs_f64());
    if let Some(items) = items {
        if let Some(rate) = phase_metrics.items_per_second {
            println!("   📈 {} itens processados ({:.0}/s)", format_number(items), rate);
        }
    }
    
    save_metrics(metrics);
}

fn main() {
    let main_start = Instant::now();
    let args = Args::parse();
    
    let mut metrics = Metrics {
        start_time: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        threads_used: args.threads.unwrap_or_else(num_cpus::get),
        gpu_used: args.gpu,
        charset_used: String::new(),
        min_length: args.min,
        max_length: args.max,
        partitions: args.parallel,
        partition_id: args.partition_id,
        ..Default::default()
    };

    println!("🔓 Universal Brute Force Tool");
    println!("=========================================\n");
    
    // Verifica se está usando comando customizado ou arquivo
    let (using_command, rar_path) = if let Some(cmd) = &args.command {
        println!("🔧 Modo comando customizado");
        println!("   Comando: {}", cmd);
        println!("   Placeholder: $password\n");
        (true, "command".to_string())
    } else if let Some(file) = &args.file {
        if !file.exists() {
            eprintln!("❌ Arquivo não encontrado: {:?}", file);
            std::process::exit(1);
        }
        (false, file.to_str().unwrap().to_string())
    } else {
        eprintln!("❌ Especifique um comando com --command");
        eprintln!("\nExemplos:");
        eprintln!("  {} --command \"mysql -u root -p$password -e quit\" --az --09 -m 4 -x 8", std::env::args().next().unwrap());
        eprintln!("  {} --command \"curl -u admin:$password http://site.com\" --custom \"abc123\" -m 6 -x 6", std::env::args().next().unwrap());
        eprintln!("  {} --command \"ssh user@host -p $password exit\" --wordlist passwords.txt", std::env::args().next().unwrap());
        std::process::exit(1);
    };

    let rar_path = rar_path.as_str();

    // Verificar GPU se solicitado
    if args.gpu {
        let gpu_info = gpu::detect_gpu();
        if gpu_info.available {
            println!("🎮 GPU detectada: {}", gpu_info.name);
            println!("💾 Memória GPU: {} GB", gpu_info.memory / (1024 * 1024 * 1024));
            println!("🎛️  Unidades de computação: {}", gpu_info.compute_units);
            
            #[cfg(feature = "gpu")]
            println!("✅ GPU habilitada para geração de senhas\n");
            
            #[cfg(not(feature = "gpu"))]
            {
                println!("⚠️  Suporte a GPU não compilado. Compile com: cargo build --release --features gpu");
                println!("💻 Continuando com processamento em CPU\n");
            }
        } else {
            println!("⚠️  GPU não disponível: {}", gpu_info.name);
            println!("💻 Continuando com processamento em CPU\n");
        }
    }

    // Configurar threads
    let available_cpus = num_cpus::get();
    let max_cpus = args.max_cpus.unwrap_or(available_cpus);
    let num_threads = args.threads.unwrap_or(max_cpus.min(available_cpus));
    
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    if using_command {
        println!("🎯 Alvo: Comando customizado");
    } else {
        println!("📁 Arquivo: {}", rar_path);
    }
    println!("💻 CPUs disponíveis: {}", available_cpus);
    println!("🔧 Usando {} threads (max permitido: {})", num_threads, max_cpus);
    
    // Configurar partições para paralelização
    if args.parallel > 1 {
        println!("🔀 Modo paralelo: {} partições", args.parallel);
        
        if let Some(id) = args.partition_id {
            if id >= args.parallel {
                eprintln!("❌ ID da partição ({}) deve ser menor que o número de partições ({})", id, args.parallel);
                std::process::exit(1);
            }
            println!("🎯 Executando partição: {}", id);
        } else {
            println!("💡 Para executar uma partição específica, use --partition-id <0-{}>", args.parallel - 1);
            
            // Mostrar exemplo de como executar em paralelo
            let charset = build_charset(&args);
            if !charset.is_empty() && args.min == args.max {
                println!("\n🚀 Exemplo de execução paralela:");
                let seeds = calculate_partition_seeds(&charset, args.min, args.parallel);
                println!("\n📝 Comandos para executar em terminais separados:");
                for i in 0..args.parallel {
                    println!("   Terminal {}: {} --partition-id {}", 
                        i + 1, 
                        std::env::args().collect::<Vec<String>>().join(" "),
                        i
                    );
                }
                println!("\n⚠️  Sem --partition-id especificado, executando todas as partições sequencialmente...");
            }
        }
    }
    println!();

    // Verifica uso de memória
    if !args.stream && !args.disk_cache && !args.wordlist.is_some() {
        let charset = build_charset(&args);
        if !charset.is_empty() {
            let mut total: usize = 0;
            let chars: Vec<char> = charset.chars().collect();
            for len in args.min..=args.max {
                total = total.saturating_add(chars.len().pow(len as u32));
            }
            
            if total > 10_000_000 {
                println!("⚠️  Aviso: {} combinações serão armazenadas em memória!", format_number(total));
                println!("💡 Considere usar --stream (sem armazenar) ou --disk-cache (salvar em disco)\n");
            }
        }
    }

    // Modo streaming
    if args.stream || args.save_progress || args.resume {
        let charset = build_charset(&args);
        if charset.is_empty() {
            eprintln!("❌ Nenhum charset selecionado!");
            eprintln!("   Use --az, --AZ, --09, --specials ou --custom");
            std::process::exit(1);
        }
        
        // Se retomando, verificar se checkpoint existe
        if args.resume && !args.checkpoint_file.exists() {
            eprintln!("❌ Checkpoint não encontrado: {:?}", args.checkpoint_file);
            eprintln!("💡 Use --save-progress para criar um checkpoint primeiro");
            std::process::exit(1);
        }
        
        let start = Instant::now();
        let found = bruteforce_streaming_with_checkpoint(
            rar_path,
            &charset,
            args.min,
            args.max,
            args.verbose,
            args.save_progress,
            &args.checkpoint_file,
            args.resume,
            args.partition_id,
            args.parallel,
            args.command.as_deref(),
        );
        let duration = start.elapsed();
        let tested_count = TESTED.load(Ordering::Relaxed);
        
        match found {
            Some(password) => {
                println!("\n🎉 =========================================");
                println!("🎉 SENHA ENCONTRADA: {}", password);
                println!("🎉 ========================================\n");
                std::fs::write("senha_encontrada.txt", password).ok();
                println!("💾 Senha salva em: senha_encontrada.txt");
            }
            None => {
                println!("\n❌ Nenhuma senha funcionou.");
                if args.save_progress {
                    println!("💾 Progresso salvo. Use --resume para continuar depois.");
                }
            }
        }
        
        println!("\n📊 Estatísticas:");
        println!("   Senhas testadas: {}", format_number(tested_count));
        println!("   Tempo decorrido: {:.2?}", duration);
        println!("   Velocidade: {:.0} senhas/seg", tested_count as f64 / duration.as_secs_f64());
        return;
    }

    let passwords = if let Some(wordlist_path) = &args.wordlist {
        // Modo wordlist
        let wordlist_start = Instant::now();
        println!("📖 Carregando wordlist: {:?}", wordlist_path);
        match std::fs::read_to_string(wordlist_path) {
            Ok(content) => {
                let passwords: Vec<String> = content.lines().map(|s| s.to_string()).collect();
                println!("📝 {} senhas carregadas\n", format_number(passwords.len()));
                metrics.total_passwords_generated = passwords.len();
                update_phase_metrics(&mut metrics, "Carregar wordlist", wordlist_start, Some(passwords.len()));
                passwords
            }
            Err(e) => {
                eprintln!("❌ Erro ao ler wordlist: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Modo força bruta
        let charset_start = Instant::now();
        let charset = build_charset(&args);
        metrics.charset_used = charset.clone();
        
        if charset.is_empty() {
            eprintln!("❌ Nenhum charset selecionado!");
            eprintln!("   Use --az, --AZ, --09, --specials ou --custom");
            eprintln!("\nExemplos:");
            eprintln!("  {} --command \"your_command $password\" --az --09 -m 4 -x 6", std::env::args().next().unwrap());
            eprintln!("  {} --command \"your_command $password\" --AZ --az --09 -m 8 -x 8", std::env::args().next().unwrap());
            eprintln!("  {} --command \"your_command $password\" --custom '123' -m 6 -x 6", std::env::args().next().unwrap());
            std::process::exit(1);
        }

        update_phase_metrics(&mut metrics, "Configurar charset", charset_start, Some(charset.len()));
        
        let gen_start = Instant::now();
        let passwords = generate_passwords_bruteforce_with_gpu(&charset, args.min, args.max, args.gpu, args.workgroup_size, args.gpu_batch_size);
        metrics.total_passwords_generated = passwords.len();
        let gen_type = if args.gpu { "Gerar senhas (GPU)" } else { "Gerar senhas (CPU)" };
        update_phase_metrics(&mut metrics, gen_type, gen_start, Some(passwords.len()));
        passwords
    };

    if passwords.is_empty() {
        return;
    }

    // Salvar dicionário se solicitado
    if let Some(dict_path) = &args.save_dictionary {
        match save_dictionary_to_file(&passwords, dict_path) {
            Ok(_) => println!("✅ Dicionário salvo em: {:?}\n", dict_path),
            Err(e) => eprintln!("❌ Erro ao salvar dicionário: {}\n", e),
        }
        
        // Se apenas gerar, parar aqui
        if args.generate_only {
            println!("🎯 Modo somente geração - não testando senhas.");
            println!("📊 Total de senhas geradas: {}", format_number(passwords.len()));
            return;
        }
    }

    let pb = ProgressBar::new(passwords.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    println!("🔍 Iniciando força bruta...\n");
    let brute_start = Instant::now();

    let found_password = if using_command {
        let command = args.command.as_ref().unwrap();
        passwords.par_iter().find_any(|password| {
            pb.inc(1);
            if args.verbose {
                pb.set_message(format!("Testando: {}", password));
            }
            test_password_with_command(command, password, args.verbose)
        })
    } else {
        passwords.par_iter().find_any(|password| {
            pb.inc(1);
            if args.verbose {
                pb.set_message(format!("Testando: {}", password));
            }
            test_password(rar_path, password)
        })
    };

    let duration = brute_start.elapsed();
    let tested_count = TESTED.load(Ordering::Relaxed);
    
    metrics.total_passwords_tested = tested_count;
    update_phase_metrics(&mut metrics, "Força bruta", brute_start, Some(tested_count));

    match found_password {
        Some(password) => {
            pb.finish_with_message(format!("✅ SENHA ENCONTRADA: {}", password));
            metrics.password_found = Some(password.to_string());
            println!("\n🎉 =========================================");
            println!("🎉 SENHA ENCONTRADA: {}", password);
            println!("🎉 ========================================\n");

            // Salvar em arquivo
            if let Err(e) = std::fs::write("senha_encontrada.txt", password) {
                eprintln!("⚠️  Erro ao salvar senha: {}", e);
            } else {
                println!("💾 Senha salva em: senha_encontrada.txt");
            }

            println!("\n📊 Estatísticas:");
            println!("   Senhas testadas: {}", format_number(tested_count));
            println!("   Tempo decorrido: {:.2?}", duration);
            println!("   Velocidade: {:.0} senhas/seg", tested_count as f64 / duration.as_secs_f64());
        }
        None => {
            pb.finish_with_message("Concluído");
            println!("\n❌ Nenhuma senha funcionou.");
            println!("\n📊 Estatísticas:");
            println!("   Senhas testadas: {}", format_number(tested_count));
            println!("   Tempo decorrido: {:.2?}", duration);
            println!("   Velocidade: {:.0} senhas/seg", tested_count as f64 / duration.as_secs_f64());
            println!("\n💡 Dicas:");
            println!("   - Tente outros charsets ou tamanhos");
            println!("   - Use uma wordlist específica");
            println!("   - Aumente o tamanho máximo da senha");
        }
    }
    
    // Finalizar e salvar métricas
    let total_duration = main_start.elapsed();
    metrics.end_time = Some(chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string());
    metrics.total_duration_seconds = total_duration.as_secs_f64();
    metrics.passwords_per_second = tested_count as f64 / total_duration.as_secs_f64();
    
    save_metrics(&metrics);
    
    println!("\n📊 RESUMO FINAL:");
    println!("   ⏱️  Tempo total: {:.2}s", total_duration.as_secs_f64());
    println!("   📝 Senhas geradas: {}", format_number(metrics.total_passwords_generated));
    println!("   ✅ Senhas testadas: {}", format_number(tested_count));
    println!("   ⚡ Velocidade média: {:.0} senhas/seg", metrics.passwords_per_second);
    println!("   💾 Métricas detalhadas salvas em: metadata.json");
}
