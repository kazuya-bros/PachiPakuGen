use base64::{engine::general_purpose::STANDARD, Engine};
use image::codecs::png::{CompressionType, FilterType as PngFilterType, PngEncoder};
use image::{DynamicImage, ExtendedColorType, ImageEncoder};
use std::io::Cursor;

/// Encode a DynamicImage as base64 PNG data URI for frontend display.
/// プレビュー用途のため圧縮率より速度を優先する（CompressionType::Fast）。
/// デフォルト圧縮はプレビュー1回で数百ms〜数秒かかり、UI応答の主なボトルネックだった。
pub fn image_to_base64_png(img: &DynamicImage) -> String {
    let rgba = img.to_rgba8();
    let mut buf = Cursor::new(Vec::new());
    let encoder = PngEncoder::new_with_quality(&mut buf, CompressionType::Fast, PngFilterType::Adaptive);
    encoder
        .write_image(
            rgba.as_raw(),
            rgba.width(),
            rgba.height(),
            ExtendedColorType::Rgba8,
        )
        .expect("Failed to encode PNG");
    let encoded = STANDARD.encode(buf.into_inner());
    format!("data:image/png;base64,{}", encoded)
}

/// Encode a DynamicImage as base64 JPEG (smaller size for previews).
pub fn image_to_base64_jpeg(img: &DynamicImage, quality: u8) -> String {
    let rgb = img.to_rgb8();
    let mut buf = Cursor::new(Vec::new());
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
    encoder.encode_image(&rgb).expect("Failed to encode JPEG");
    let encoded = STANDARD.encode(buf.into_inner());
    format!("data:image/jpeg;base64,{}", encoded)
}
