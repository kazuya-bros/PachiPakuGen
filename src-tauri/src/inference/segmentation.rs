use crate::error::AppError;
use image::{DynamicImage, GrayImage};
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

/// Run UNet segmentation and return the class map (H, W) as u8.
///
/// Input: RGB image of any size (resized internally to 512x512)
/// Output: class map where each pixel is a class ID (0-6)
///   0=BG, 1=Hair, 2=Eye, 3=Mouth, 4=Face, 5=Skin, 6=Clothes
pub fn run_segmentation(
    session: &mut Session,
    image: &DynamicImage,
) -> Result<Vec<u8>, AppError> {
    let (orig_w, orig_h) = (image.width(), image.height());

    // Resize to 512x512 (matching Python: T.Resize(512))
    let resized = image.resize_exact(512, 512, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    // Convert to (1, 3, 512, 512) float32 tensor, scale to [0, 1]
    let mut input = Array4::<f32>::zeros([1, 3, 512, 512]);
    for y in 0..512u32 {
        for x in 0..512u32 {
            let pixel = rgb.get_pixel(x, y);
            input[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            input[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            input[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }
    }

    // Run inference
    let input_tensor = TensorRef::from_array_view(&input)?;
    let outputs = session.run(ort::inputs!["input" => input_tensor])?;
    let output = outputs["output"].try_extract_array::<f32>()?;

    // output shape: (1, 7, 512, 512) — softmax probabilities
    // argmax across channel dim to get class map
    let mut seg_map_512 = vec![0u8; 512 * 512];
    for y in 0..512usize {
        for x in 0..512usize {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_class = 0u8;
            for c in 0..7usize {
                let val = output[[0, c, y, x]];
                if val > max_val {
                    max_val = val;
                    max_class = c as u8;
                }
            }
            seg_map_512[y * 512 + x] = max_class;
        }
    }

    // Resize seg map back to original size (nearest-neighbor)
    let seg_gray = GrayImage::from_raw(512, 512, seg_map_512)
        .ok_or_else(|| AppError::General("Failed to create seg map image".into()))?;
    let seg_resized = image::imageops::resize(
        &seg_gray,
        orig_w,
        orig_h,
        image::imageops::FilterType::Nearest,
    );

    Ok(seg_resized.into_raw())
}

/// Extract eye and mouth binary masks from segmentation map.
/// Returns (eye_mask, mouth_mask) as Vec<u8> with values 0-255.
pub fn extract_masks(
    seg_map: &[u8],
    width: u32,
    height: u32,
) -> (Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;

    // Binary masks: class 2 = eye, class 3 = mouth
    let mut eye_mask = vec![0u8; w * h];
    let mut mouth_mask = vec![0u8; w * h];

    for i in 0..(w * h) {
        if seg_map[i] == 2 {
            eye_mask[i] = 255;
        }
        if seg_map[i] == 3 {
            mouth_mask[i] = 255;
        }
    }

    eye_mask = postprocess_mask(&eye_mask, w, h);
    mouth_mask = postprocess_mask(&mouth_mask, w, h);

    (eye_mask, mouth_mask)
}

/// Post-process a binary mask: dilate + gaussian blur for smooth boundaries.
pub fn postprocess_mask(mask: &[u8], w: usize, h: usize) -> Vec<u8> {
    let dilated = dilate(mask, w, h, 2, 2);
    gaussian_blur_mask(&dilated, w, h, 3)
}

/// Simple dilation with a circular-ish kernel of given radius.
pub fn dilate(mask: &[u8], w: usize, h: usize, radius: i32, iterations: u32) -> Vec<u8> {
    let mut current = mask.to_vec();
    for _ in 0..iterations {
        let mut next = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut max_val = 0u8;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        if dx * dx + dy * dy > radius * radius + radius {
                            continue;
                        }
                        let ny = y as i32 + dy;
                        let nx = x as i32 + dx;
                        if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                            let val = current[ny as usize * w + nx as usize];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }
                }
                next[y * w + x] = max_val;
            }
        }
        current = next;
    }
    current
}

/// Simple box-approximated gaussian blur for mask smoothing.
pub fn gaussian_blur_mask(mask: &[u8], w: usize, h: usize, radius: usize) -> Vec<u8> {
    let gray = GrayImage::from_raw(w as u32, h as u32, mask.to_vec())
        .expect("Invalid mask dimensions");
    let blurred = imageproc::filter::gaussian_blur_f32(&gray, radius as f32);
    blurred.into_raw()
}

/// Apply a mask to an RGBA image: keep pixels where mask > 0, clear others.
/// The mask is used as alpha blending: result_alpha = min(original_alpha, mask_value).
pub fn apply_mask_to_rgba(
    image: &DynamicImage,
    mask: &[u8],
    width: u32,
    height: u32,
) -> DynamicImage {
    let rgba = image.to_rgba8();
    let mut result = image::RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let mask_val = mask[idx];
            if mask_val > 0 {
                let p = rgba.get_pixel(x, y);
                let alpha = p[3].min(mask_val);
                result.put_pixel(x, y, image::Rgba([p[0], p[1], p[2], alpha]));
            }
        }
    }

    DynamicImage::ImageRgba8(result)
}

/// Invert mask: keep pixels where mask == 0, clear where mask > 0.
pub fn apply_inverted_mask_to_rgba(
    image: &DynamicImage,
    mask: &[u8],
    width: u32,
    height: u32,
) -> DynamicImage {
    let rgba = image.to_rgba8();
    let mut result = image::RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let mask_val = mask[idx];
            if mask_val < 255 {
                let p = rgba.get_pixel(x, y);
                let keep = 255 - mask_val;
                let alpha = p[3].min(keep);
                result.put_pixel(x, y, image::Rgba([p[0], p[1], p[2], alpha]));
            }
        }
    }

    DynamicImage::ImageRgba8(result)
}

/// Create an overlay visualization: eye=blue, mouth=red, eyebrow=green.
/// All masks are optional — only present masks are rendered.
/// Returns RGB DynamicImage.
pub fn create_overlay(
    base: &DynamicImage,
    eye_mask: Option<&[u8]>,
    mouth_mask: Option<&[u8]>,
    eyebrow_mask: Option<&[u8]>,
    width: u32,
    height: u32,
) -> DynamicImage {
    let rgb = base.to_rgb8();
    let mut overlay = rgb.clone();
    let alpha = 0.5f32;

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let pixel = overlay.get_pixel_mut(x, y);
            let mut rf = pixel.0[0] as f32;
            let mut gf = pixel.0[1] as f32;
            let mut bf = pixel.0[2] as f32;

            // Eye mask: blue tint
            if let Some(eye) = eye_mask {
                let eye_a = eye[idx] as f32 / 255.0 * alpha;
                rf = rf * (1.0 - eye_a);
                gf = gf * (1.0 - eye_a);
                bf = bf * (1.0 - eye_a) + 255.0 * eye_a;
            }

            // Mouth mask: red tint
            if let Some(mouth) = mouth_mask {
                let mouth_a = mouth[idx] as f32 / 255.0 * alpha;
                rf = rf * (1.0 - mouth_a) + 255.0 * mouth_a;
                gf = gf * (1.0 - mouth_a);
                bf = bf * (1.0 - mouth_a);
            }

            // Eyebrow mask: green tint
            if let Some(eb_mask) = eyebrow_mask {
                let eb_a = eb_mask[idx] as f32 / 255.0 * alpha;
                rf = rf * (1.0 - eb_a);
                gf = gf * (1.0 - eb_a) + 255.0 * eb_a;
                bf = bf * (1.0 - eb_a);
            }

            pixel.0 = [
                rf.clamp(0.0, 255.0) as u8,
                gf.clamp(0.0, 255.0) as u8,
                bf.clamp(0.0, 255.0) as u8,
            ];
        }
    }

    DynamicImage::ImageRgb8(overlay)
}
