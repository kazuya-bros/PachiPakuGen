use image::DynamicImage;
use ort::session::Session;
use std::collections::HashMap;
use std::sync::Mutex;

pub struct AppState {
    pub rife_session: Mutex<Option<Session>>,

    // Per-slot imported layers from See-Through (PSD or folder).
    // Key "current": the most recently loaded PSD/folder's layers
    pub slot_layers: Mutex<HashMap<String, HashMap<String, DynamicImage>>>,

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

    // Cached neck image extracted from original via SAM3
    pub cached_neck: Mutex<Option<DynamicImage>>,

    // Cached SAM3 mouth mask (grayscale, from base PSD, reused for all diffs)
    pub cached_mouth_mask: Mutex<Option<Vec<u8>>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            rife_session: Mutex::new(None),
            slot_layers: Mutex::new(HashMap::new()),
            layer_mapping: Mutex::new(HashMap::new()),
            parts: Mutex::new(HashMap::new()),
            canvas_width: Mutex::new(0),
            canvas_height: Mutex::new(0),
            cached_original: Mutex::new(None),
            cached_neck: Mutex::new(None),
            cached_mouth_mask: Mutex::new(None),
        }
    }
}
