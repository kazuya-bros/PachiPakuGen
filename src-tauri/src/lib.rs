mod commands;
mod error;
mod inference;
mod processing;
mod state;

use state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[cfg(target_os = "windows")]
    if let Ok(store) = windows_native_keyring_store::Store::new() {
        keyring_core::set_default_store(store);
    }

    // Initialize ONNX Runtime
    let ok = ort::init().with_name("PachiPakuGen").commit();
    eprintln!(
        "[PachiPakuGen] ORT init: {}",
        if ok { "success" } else { "already initialized" }
    );

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::parts::load_slot,
            commands::parts::create_base,
            commands::parts::get_base_preview,
            commands::parts::get_mapping_preview,
            commands::parts::render_category,
            commands::parts::get_all_layers_preview,
            commands::see_through::load_expression_source_preview,
            commands::see_through::get_see_through_runtime_status,
            commands::see_through::prepare_see_through_runtime,
            commands::see_through::start_see_through_model_download,
            commands::motion_lab::save_motion_loop_frame,
            commands::motion_lab::finalize_motion_loop_export,
            commands::see_through::run_see_through,
            commands::see_through::probe_see_through_layers,
            commands::see_through::cancel_see_through,
            commands::see_through::list_see_through_gpus,
            commands::see_through::set_see_through_gpu,
            commands::see_through::get_see_through_install_location,
            commands::see_through::set_see_through_install_location,
            commands::see_through::get_hf_token_status,
            commands::see_through::save_hf_token,
            commands::see_through::delete_hf_token,
            commands::workspace::create_expression_workspace,
            commands::workspace::load_expression_workspace,
            commands::workspace::prepare_workspace_codex_request,
            commands::workspace::inspect_workspace_generated_parts,
            commands::workspace::update_expression_workspace_step,
            commands::workspace::regress_expression_workspace_step,
            commands::expression::get_expression_api_status,
            commands::expression::save_expression_api_key,
            commands::expression::delete_expression_api_key,
            commands::expression::prepare_codex_expression_job,
            commands::expression::inspect_codex_generated_parts,
            commands::expression::load_codex_expression_job,
            commands::expression::extract_codex_generated_parts,
            commands::expression::cache_codex_source_see_through,
            commands::expression::load_codex_source_see_through,
            commands::expression::preview_codex_composite,
            commands::expression::generate_codex_rife_outputs,
            commands::expression::save_codex_base_parts,
            commands::expression::save_base_editor_state,
            commands::expression::load_base_editor_state,
            commands::expression::adjust_codex_extracted_parts_batch,
            commands::expression::generate_expression_set,
            commands::motion_lab::load_motion_lab_parts,
            commands::motion_lab::save_motion_lab_manifest,
            commands::motion_lab::load_motion_lab_manifest,
            commands::motion_lab::load_spritalk_motion_profile,
            commands::motion_lab::save_spritalk_motion_profile,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
