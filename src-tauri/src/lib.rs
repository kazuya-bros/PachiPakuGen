mod commands;
mod error;
mod inference;
mod processing;
mod state;

use state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize ONNX Runtime
    let ok = ort::init()
        .with_name("PachiPakuGen")
        .commit();
    eprintln!("[PachiPakuGen] ORT init: {}", if ok { "success" } else { "already initialized" });

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::parts::load_slot,
            commands::parts::create_base,
            commands::parts::create_diff,
            commands::parts::load_original_image,
            commands::parts::get_base_preview,
            commands::parts::get_mapping_preview,
            commands::parts::render_category,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
