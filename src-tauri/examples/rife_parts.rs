#[path = "../src/processing/composite.rs"]
mod composite;
#[path = "../src/error.rs"]
mod error;
#[path = "../src/inference/rife.rs"]
mod rife;
#[allow(dead_code)]
#[path = "../src/inference/session.rs"]
mod session;

use composite::{extract_part_from_body_composite, premultiply_onto_body};
use error::AppError;
use image::DynamicImage;
use rife::rife_interpolate;
use session::create_session;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn usage() -> &'static str {
    "Usage: cargo run --example rife_parts -- <model.onnx> <base.png> <part-a.png> <part-b.png> <output-dir> [frame-count]"
}

fn load_rgba(path: &Path, width: u32, height: u32) -> Result<image::RgbaImage, AppError> {
    let image = image::open(path)?;
    if image.width() != width || image.height() != height {
        return Err(AppError::General(format!(
            "Canvas mismatch: {} is {}x{}, expected {}x{}",
            path.display(),
            image.width(),
            image.height(),
            width,
            height
        )));
    }
    Ok(image.to_rgba8())
}

fn main() -> Result<(), AppError> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 6 || args.len() > 7 {
        return Err(AppError::General(usage().into()));
    }
    let model = PathBuf::from(&args[1]);
    let base_path = PathBuf::from(&args[2]);
    let part_a_path = PathBuf::from(&args[3]);
    let part_b_path = PathBuf::from(&args[4]);
    let output_dir = PathBuf::from(&args[5]);
    let frame_count: u32 = args
        .get(6)
        .map(|value| value.parse())
        .transpose()
        .map_err(|_| AppError::General("frame-count must be an integer".into()))?
        .unwrap_or(8);
    if !(2..=30).contains(&frame_count) {
        return Err(AppError::General(
            "frame-count must be between 2 and 30".into(),
        ));
    }

    let base = image::open(&base_path)?;
    let width = base.width();
    let height = base.height();
    let part_a = load_rgba(&part_a_path, width, height)?;
    let part_b = load_rgba(&part_b_path, width, height)?;
    let base_rgb = base.to_rgb8();
    let rife_a = premultiply_onto_body(&base_rgb, &part_a, width, height);
    let rife_b = premultiply_onto_body(&base_rgb, &part_b, width, height);

    let _ = ort::init()
        .with_name("PachiPakuGen RIFE prototype")
        .commit();
    let mut session = create_session(&model)?;
    fs::create_dir_all(&output_dir)?;
    let started = Instant::now();

    for index in 0..frame_count {
        let ratio = index as f32 / (frame_count - 1) as f32;
        let frame = if index == 0 {
            DynamicImage::ImageRgba8(part_a.clone())
        } else if index + 1 == frame_count {
            DynamicImage::ImageRgba8(part_b.clone())
        } else {
            let interpolated = rife_interpolate(&mut session, &rife_a, &rife_b, ratio)?;
            extract_part_from_body_composite(
                &interpolated,
                &base_rgb,
                &part_a,
                &part_b,
                ratio,
                width,
                height,
            )
        };
        frame.save(output_dir.join(format!("{:03}.png", index + 1)))?;
    }

    println!(
        "Generated {} frames at {}x{} in {:.2}s: {}",
        frame_count,
        width,
        height,
        started.elapsed().as_secs_f32(),
        output_dir.display()
    );
    Ok(())
}
