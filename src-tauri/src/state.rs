use image::DynamicImage;
use ort::session::Session;
use std::collections::HashMap;
use std::sync::Mutex;

/// A named pair of part images for RIFE interpolation.
/// e.g. ("eye", eye_open, eye_closed) or ("mouth_a", mouth_closed, mouth_a)
#[derive(Clone)]
pub struct PartPair {
    pub name: String,           // e.g. "eye", "mouth_a", "mouth_i", ...
    pub image_a: DynamicImage,  // Start frame (e.g. open, closed)
    pub image_b: DynamicImage,  // End frame
}

/// Generated frames for a single part pair.
pub struct PartFrames {
    pub name: String,
    pub frames: Vec<DynamicImage>,
}

pub struct AppState {
    pub rife_session: Mutex<Option<Session>>,
    pub seg_session: Mutex<Option<Session>>,    // UNet segmentation session (fallback)

    // Input images as loaded (before separation)
    // Keys: "eye_open", "eye_closed", "mouth_closed", "mouth_a", ... "body", "hair"
    pub raw_images: Mutex<HashMap<String, DynamicImage>>,

    // Separated parts (after mask extraction)
    // Keys same as raw_images but images have only eye or mouth region
    pub parts: Mutex<HashMap<String, DynamicImage>>,

    // Raw masks (from SAM3/UNet, before brush editing)
    pub raw_eye_mask: Mutex<Option<Vec<u8>>>,
    pub raw_mouth_mask: Mutex<Option<Vec<u8>>>,
    pub raw_eyebrow_mask: Mutex<Option<Vec<u8>>>,

    // Adjusted masks (after brush editing; initially same as raw)
    pub eye_mask: Mutex<Option<Vec<u8>>>,
    pub mouth_mask: Mutex<Option<Vec<u8>>>,
    pub eyebrow_mask: Mutex<Option<Vec<u8>>>,

    // Combined mask (all parts merged)
    pub combined_mask: Mutex<Option<Vec<u8>>>,
    pub mask_width: Mutex<u32>,
    pub mask_height: Mutex<u32>,

    // Computed part pairs for RIFE processing
    pub part_pairs: Mutex<Vec<PartPair>>,

    // Generated frames per pair
    pub generated: Mutex<Vec<PartFrames>>,

    // Frame count setting
    pub frame_count: Mutex<u32>,

    // Canvas dimensions (from body part)
    pub canvas_width: Mutex<u32>,
    pub canvas_height: Mutex<u32>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            rife_session: Mutex::new(None),
            seg_session: Mutex::new(None),
            raw_images: Mutex::new(HashMap::new()),
            parts: Mutex::new(HashMap::new()),
            raw_eye_mask: Mutex::new(None),
            raw_mouth_mask: Mutex::new(None),
            raw_eyebrow_mask: Mutex::new(None),
            eye_mask: Mutex::new(None),
            mouth_mask: Mutex::new(None),
            eyebrow_mask: Mutex::new(None),
            combined_mask: Mutex::new(None),
            mask_width: Mutex::new(0),
            mask_height: Mutex::new(0),
            part_pairs: Mutex::new(Vec::new()),
            generated: Mutex::new(Vec::new()),
            frame_count: Mutex::new(4),
            canvas_width: Mutex::new(0),
            canvas_height: Mutex::new(0),
        }
    }
}
