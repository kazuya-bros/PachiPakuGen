use image::{DynamicImage, GrayImage};
use ort::session::Session;
use std::collections::HashMap;
use std::sync::Mutex;

pub struct AppState {
    pub rife_session: Mutex<Option<Session>>,

    // Per-slot imported layers from See-Through (PSD or folder).
    // Key "current": the most recently loaded PSD/folder's layers
    pub slot_layers: Mutex<HashMap<String, HashMap<String, DynamicImage>>>,

    // Current See-Through PSD layer order, bottom/back first. HashMap alone
    // cannot preserve this order, which is required for recomposition.
    pub slot_layer_order: Mutex<Vec<String>>,

    // Per-pixel See-Through depth maps for the current PSD. Lower values are
    // closer to the viewer and allow local visibility clipping at overlaps.
    pub slot_depth_maps: Mutex<HashMap<String, GrayImage>>,

    // User-confirmed mapping for adjustable layers.
    pub layer_mapping: Mutex<HashMap<String, String>>,

    // Merged parts (base body + base eye/mouth frames)
    // Keys: "body", "hair", "hair_back", "eye_open"/"eye_closed", "mouth_closed"/etc.
    pub parts: Mutex<HashMap<String, DynamicImage>>,

    // Canvas dimensions (from body part)
    pub canvas_width: Mutex<u32>,
    pub canvas_height: Mutex<u32>,

    // Cached original image (resized to canvas dimensions)
    pub cached_original: Mutex<Option<DynamicImage>>,

    // Cached SAM3 mouth mask (grayscale, from base PSD, reused for all diffs)
    pub cached_mouth_mask: Mutex<Option<Vec<u8>>>,

    // Raw SAM3 mouth mask before UI dilation/blur adjustments.
    pub cached_mouth_raw_mask: Mutex<Option<Vec<u8>>>,

    // Per-original cache so UI adjustments do not rerun SAM3 for vowel previews.
    pub cached_mouth_originals: Mutex<HashMap<String, DynamicImage>>,
    pub cached_mouth_raw_masks: Mutex<HashMap<String, Vec<u8>>>,

    // PID of the currently running See-Through setup or inference process.
    pub see_through_pid: Mutex<Option<u32>>,

    // Serializes runtime checkout/patching and inference. The UI also disables concurrent
    // actions, but this backend guard prevents races from direct command invocation.
    pub see_through_runtime_lock: Mutex<()>,

    // PID of the user-visible console used only for the large model pre-download.
    // Kept separate so setup/inference cancellation cannot kill a user-managed download.
    pub see_through_model_download_pid: Mutex<Option<u32>>,

    // ユーザーが選択したSee-Through実行GPU（nvidia-smiのindex）。None=最大VRAMを自動選択
    pub see_through_gpu_index: Mutex<Option<u32>>,

    // Step4のunifiedレイヤー順から導出したグループ間の描画順（背面→前面）。
    // 空 = 従来の固定z順。save_codex_base_parts が layer-order.json として出力する。
    pub base_layer_group_order: Mutex<Vec<String>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            rife_session: Mutex::new(None),
            slot_layers: Mutex::new(HashMap::new()),
            slot_layer_order: Mutex::new(Vec::new()),
            slot_depth_maps: Mutex::new(HashMap::new()),
            layer_mapping: Mutex::new(HashMap::new()),
            parts: Mutex::new(HashMap::new()),
            canvas_width: Mutex::new(0),
            canvas_height: Mutex::new(0),
            cached_original: Mutex::new(None),
            cached_mouth_mask: Mutex::new(None),
            cached_mouth_raw_mask: Mutex::new(None),
            cached_mouth_originals: Mutex::new(HashMap::new()),
            cached_mouth_raw_masks: Mutex::new(HashMap::new()),
            see_through_pid: Mutex::new(None),
            see_through_runtime_lock: Mutex::new(()),
            see_through_model_download_pid: Mutex::new(None),
            see_through_gpu_index: Mutex::new(None),
            base_layer_group_order: Mutex::new(Vec::new()),
        }
    }
}
