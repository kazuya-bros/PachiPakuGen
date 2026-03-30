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
            commands::parts::load_part_preview,
            commands::parts::load_parts,
            commands::segmentation::check_model_availability,
            commands::segmentation::run_segmentation_cmd,
            commands::segmentation::apply_masks_and_build_pairs,
            commands::segmentation::get_brush_background_cmd,
            commands::segmentation::get_target_mask_cmd,
            commands::segmentation::apply_brush_mask_cmd,
            commands::segmentation::delete_mask_cmd,
            commands::generation::generate_frames,
            commands::generation::get_composite_preview,
            commands::export::export_frames,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
