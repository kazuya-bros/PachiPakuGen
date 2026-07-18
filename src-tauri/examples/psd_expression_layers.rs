use image::{DynamicImage, RgbaImage};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const EYE_NAMES: &[&str] = &["eyewhite", "irides", "eyelash", "eyebrow"];
const HAIR_NAMES: &[&str] = &["front_hair", "headwear"];

fn usage() -> &'static str {
    "Usage: cargo run --example psd_expression_layers -- <input.psd> <output-dir>"
}

fn normalize_layer_name(name: &str) -> String {
    name.to_lowercase().replace(' ', "_")
}

fn base_layer_name(name: &str) -> &str {
    name.strip_suffix("-l")
        .or_else(|| name.strip_suffix("-r"))
        .or_else(|| name.strip_suffix("_l"))
        .or_else(|| name.strip_suffix("_r"))
        .unwrap_or(name)
}

fn alpha_bounds(image: &RgbaImage) -> Option<(u32, u32, u32, u32, u64)> {
    let mut left = image.width();
    let mut top = image.height();
    let mut right = 0;
    let mut bottom = 0;
    let mut pixels = 0u64;
    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel[3] == 0 {
            continue;
        }
        left = left.min(x);
        top = top.min(y);
        right = right.max(x + 1);
        bottom = bottom.max(y + 1);
        pixels += 1;
    }
    (pixels > 0).then_some((left, top, right, bottom, pixels))
}

fn alpha_composite_onto(base: &mut RgbaImage, layer: &RgbaImage) {
    for (x, y, foreground) in layer.enumerate_pixels() {
        let fa = foreground[3] as f32 / 255.0;
        if fa <= 0.0 {
            continue;
        }
        let background = base.get_pixel(x, y);
        let ba = background[3] as f32 / 255.0;
        let out_a = fa + ba * (1.0 - fa);
        let channel = |index: usize| {
            if out_a <= 0.0 {
                0
            } else {
                (((foreground[index] as f32 * fa) + (background[index] as f32 * ba * (1.0 - fa)))
                    / out_a)
                    .round()
                    .clamp(0.0, 255.0) as u8
            }
        };
        base.put_pixel(
            x,
            y,
            image::Rgba([channel(0), channel(1), channel(2), (out_a * 255.0) as u8]),
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err(usage().into());
    }
    let input = Path::new(&args[1]);
    let output_dir = PathBuf::from(&args[2]);
    fs::create_dir_all(&output_dir)?;

    let bytes = fs::read(input)?;
    let psd = psd::Psd::from_bytes(&bytes).map_err(|error| format!("{error:?}"))?;
    let width = psd.width();
    let height = psd.height();
    let mut eye_layers = BTreeMap::new();
    let mut hair_layers = BTreeMap::new();
    let mut manifest = Vec::new();

    for layer in psd.layers() {
        let name = normalize_layer_name(layer.name());
        let Some(canvas) = RgbaImage::from_raw(width, height, layer.rgba()) else {
            continue;
        };
        let Some((left, top, right, bottom, pixels)) = alpha_bounds(&canvas) else {
            continue;
        };
        let base_name = base_layer_name(&name);
        let category = if EYE_NAMES.contains(&base_name) {
            eye_layers.insert(name.clone(), canvas.clone());
            "eye"
        } else if HAIR_NAMES.contains(&base_name) {
            hair_layers.insert(name.clone(), canvas.clone());
            "hair"
        } else {
            "other"
        };
        manifest.push(format!(
            "{name}\t{category}\talpha={pixels}\tbbox={left},{top},{right},{bottom}"
        ));
    }

    let mut eye = RgbaImage::new(width, height);
    for base_name in EYE_NAMES {
        for (name, layer) in &eye_layers {
            if base_layer_name(name) == *base_name {
                alpha_composite_onto(&mut eye, layer);
            }
        }
    }
    let mut hair = RgbaImage::new(width, height);
    for (_, layer) in hair_layers {
        alpha_composite_onto(&mut hair, &layer);
    }

    DynamicImage::ImageRgba8(eye).save(output_dir.join("eyes.png"))?;
    DynamicImage::ImageRgba8(hair).save(output_dir.join("front-hair.png"))?;
    fs::write(output_dir.join("layers.txt"), manifest.join("\n"))?;
    println!(
        "Extracted {} layers from {}x{} PSD: {}",
        manifest.len(),
        width,
        height,
        output_dir.display()
    );
    Ok(())
}
