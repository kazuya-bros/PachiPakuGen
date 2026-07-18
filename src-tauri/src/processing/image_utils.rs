use base64::{engine::general_purpose::STANDARD, Engine};
use image::codecs::png::{CompressionType, FilterType as PngFilterType, PngEncoder};
use image::{DynamicImage, ExtendedColorType, ImageEncoder, Rgba, RgbaImage};
use std::io::Cursor;

/// Encode a DynamicImage as base64 PNG data URI for frontend display.
/// プレビュー用途のため圧縮率より速度を優先する（CompressionType::Fast）。
/// デフォルト圧縮はプレビュー1回で数百ms〜数秒かかり、UI応答の主なボトルネックだった。
pub fn image_to_base64_png(img: &DynamicImage) -> String {
    let rgba = img.to_rgba8();
    let mut buf = Cursor::new(Vec::new());
    let encoder =
        PngEncoder::new_with_quality(&mut buf, CompressionType::Fast, PngFilterType::Adaptive);
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

/// 画像alphaからmask alphaを差し引く。半透明素材を同じmaskで消す場合も縁を残さない。
pub fn subtract_alpha_mask(image: &DynamicImage, mask: &DynamicImage) -> DynamicImage {
    let mut output = image.to_rgba8();
    let mask = if mask.width() == output.width() && mask.height() == output.height() {
        mask.to_rgba8()
    } else {
        mask.resize_exact(
            output.width(),
            output.height(),
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgba8()
    };
    for (pixel, mask_pixel) in output.pixels_mut().zip(mask.pixels()) {
        pixel[3] = pixel[3].saturating_sub(mask_pixel[3]);
    }
    DynamicImage::ImageRgba8(output)
}

/// 旧目フレームに焼き込まれた眉を除去する二値mask。
/// 1280px素材で必要最小だった6pxを基準に解像度比例し、近接する睫毛へ届かない10pxで上限を切る。
pub fn eyebrow_cleanup_mask(mask: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    let resized = if mask.width() == width && mask.height() == height {
        mask.to_rgba8()
    } else {
        mask.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
            .to_rgba8()
    };
    let max_dimension = u64::from(width.max(height));
    let radius = ((max_dimension * 6 + 640) / 1280).min(10) as i32;
    let mut expanded = RgbaImage::new(width, height);
    for (x, y, pixel) in resized.enumerate_pixels() {
        if pixel[3] <= 8 {
            continue;
        }
        let x = x as i32;
        let y = y as i32;
        for target_y in (y - radius).max(0)..=(y + radius).min(height as i32 - 1) {
            for target_x in (x - radius).max(0)..=(x + radius).min(width as i32 - 1) {
                expanded.put_pixel(target_x as u32, target_y as u32, Rgba([0, 0, 0, 255]));
            }
        }
    }
    DynamicImage::ImageRgba8(expanded)
}
