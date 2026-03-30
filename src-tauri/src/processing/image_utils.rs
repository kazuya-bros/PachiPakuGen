use base64::{engine::general_purpose::STANDARD, Engine};
use image::DynamicImage;
use std::io::Cursor;

/// Encode a DynamicImage as base64 PNG data URI for frontend display.
pub fn image_to_base64_png(img: &DynamicImage) -> String {
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)
        .expect("Failed to encode PNG");
    let encoded = STANDARD.encode(buf.into_inner());
    format!("data:image/png;base64,{}", encoded)
}

/// Encode a DynamicImage as base64 JPEG (smaller size for previews).
pub fn image_to_base64_jpeg(img: &DynamicImage, quality: u8) -> String {
    let rgb = img.to_rgb8();
    let mut buf = Cursor::new(Vec::new());
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
    encoder
        .encode_image(&rgb)
        .expect("Failed to encode JPEG");
    let encoded = STANDARD.encode(buf.into_inner());
    format!("data:image/jpeg;base64,{}", encoded)
}
