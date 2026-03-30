use crate::error::AppError;
use image::DynamicImage;
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

/// Run RIFE interpolation between two images at the given ratio.
///
/// ratio: 0.0 = img0, 1.0 = img1
/// Images must be the same dimensions.
pub fn rife_interpolate(
    session: &mut Session,
    img0: &DynamicImage,
    img1: &DynamicImage,
    ratio: f32,
) -> Result<DynamicImage, AppError> {
    // Return endpoints directly
    if ratio <= 0.0 {
        return Ok(img0.clone());
    }
    if ratio >= 1.0 {
        return Ok(img1.clone());
    }

    let rgb0 = img0.to_rgb8();
    let rgb1 = img1.to_rgb8();
    let (w, h) = (rgb0.width(), rgb0.height());

    // Pad to multiple of 64 (RIFE internal padding uses 16x scale with 4-block cascade)
    let ph = ((h as usize - 1) / 64 + 1) * 64;
    let pw = ((w as usize - 1) / 64 + 1) * 64;

    // Convert to (1, 3, ph, pw) float32 tensors, RGB, [0, 1]
    let mut t0 = Array4::<f32>::zeros([1, 3, ph, pw]);
    let mut t1 = Array4::<f32>::zeros([1, 3, ph, pw]);

    for y in 0..h as usize {
        for x in 0..w as usize {
            let p0 = rgb0.get_pixel(x as u32, y as u32);
            let p1 = rgb1.get_pixel(x as u32, y as u32);
            t0[[0, 0, y, x]] = p0[0] as f32 / 255.0;
            t0[[0, 1, y, x]] = p0[1] as f32 / 255.0;
            t0[[0, 2, y, x]] = p0[2] as f32 / 255.0;
            t1[[0, 0, y, x]] = p1[0] as f32 / 255.0;
            t1[[0, 1, y, x]] = p1[1] as f32 / 255.0;
            t1[[0, 2, y, x]] = p1[2] as f32 / 255.0;
        }
    }

    // Timestep as (1, 1, 1, 1) tensor
    let timestep = Array4::<f32>::from_elem([1, 1, 1, 1], ratio);

    // Create TensorRef from arrays
    let tensor0 = TensorRef::from_array_view(&t0)?;
    let tensor1 = TensorRef::from_array_view(&t1)?;
    let tensor_ts = TensorRef::from_array_view(&timestep)?;

    // Run inference
    let outputs = session.run(ort::inputs![
        "img0" => tensor0,
        "img1" => tensor1,
        "timestep" => tensor_ts
    ])?;

    let output = outputs["output"].try_extract_array::<f32>()?;

    // Convert output (1, 3, ph, pw) back to DynamicImage, crop to original size
    let mut result = image::RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = (output[[0, 0, y as usize, x as usize]] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (output[[0, 1, y as usize, x as usize]] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (output[[0, 2, y as usize, x as usize]] * 255.0).clamp(0.0, 255.0) as u8;
            result.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }

    Ok(DynamicImage::ImageRgb8(result))
}
