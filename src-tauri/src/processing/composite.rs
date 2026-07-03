use image::DynamicImage;

/// Alpha-composite layers bottom to top: body -> part (with alpha) -> hair.
/// part_rgb is the RIFE interpolation output (RGB) or original part image.
/// part_alpha is the alpha channel (w*h bytes).
/// Returns RGBA image.
pub fn alpha_composite_layers(
    body: &DynamicImage,
    part_rgb: &DynamicImage,
    part_alpha: &[u8],
    hair: &DynamicImage,
    width: u32,
    height: u32,
) -> DynamicImage {
    let body_rgba = body.to_rgba8();
    let part_rgb8 = part_rgb.to_rgb8();
    let hair_rgba = hair.to_rgba8();
    let mut result = image::RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;

            // Start with body pixel
            let bp = body_rgba.get_pixel(x, y);
            let mut r = bp[0] as f32;
            let mut g = bp[1] as f32;
            let mut b = bp[2] as f32;
            let mut a = bp[3] as f32;

            // Composite part on top
            let pp = part_rgb8.get_pixel(x, y);
            let pa = part_alpha[idx] as f32 / 255.0;
            r = r * (1.0 - pa) + pp[0] as f32 * pa;
            g = g * (1.0 - pa) + pp[1] as f32 * pa;
            b = b * (1.0 - pa) + pp[2] as f32 * pa;
            a = a * (1.0 - pa) + 255.0 * pa;

            // Composite hair on top
            let hp = hair_rgba.get_pixel(x, y);
            let ha = hp[3] as f32 / 255.0;
            r = r * (1.0 - ha) + hp[0] as f32 * ha;
            g = g * (1.0 - ha) + hp[1] as f32 * ha;
            b = b * (1.0 - ha) + hp[2] as f32 * ha;
            a = a * (1.0 - ha) + 255.0 * ha;

            result.put_pixel(
                x,
                y,
                image::Rgba([
                    r.clamp(0.0, 255.0) as u8,
                    g.clamp(0.0, 255.0) as u8,
                    b.clamp(0.0, 255.0) as u8,
                    a.clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    DynamicImage::ImageRgba8(result)
}

/// Composite a part RGBA layer onto a body RGB image, producing an opaque RGB image.
/// Transparent pixels in the part layer show the body underneath.
/// This ensures RIFE never sees black pixels from transparency.
pub fn premultiply_onto_body(
    body_rgb: &image::RgbImage,
    part_rgba: &image::RgbaImage,
    width: u32,
    height: u32,
) -> DynamicImage {
    let mut result = body_rgb.clone();
    for y in 0..height {
        for x in 0..width {
            let fp = part_rgba.get_pixel(x, y);
            let fa = fp[3] as f32 / 255.0;
            if fa > 0.0 {
                let bp = body_rgb.get_pixel(x, y);
                let r = bp[0] as f32 * (1.0 - fa) + fp[0] as f32 * fa;
                let g = bp[1] as f32 * (1.0 - fa) + fp[1] as f32 * fa;
                let b = bp[2] as f32 * (1.0 - fa) + fp[2] as f32 * fa;
                result.put_pixel(
                    x,
                    y,
                    image::Rgb([
                        r.clamp(0.0, 255.0) as u8,
                        g.clamp(0.0, 255.0) as u8,
                        b.clamp(0.0, 255.0) as u8,
                    ]),
                );
            }
        }
    }
    DynamicImage::ImageRgb8(result)
}

/// Extract a transparent part from an opaque RIFE frame using interpolated endpoint alpha.
pub fn extract_part_with_blended_alpha(
    rife_output: &DynamicImage,
    img_a_rgba: &image::RgbaImage,
    img_b_rgba: &image::RgbaImage,
    ratio: f32,
    width: u32,
    height: u32,
) -> DynamicImage {
    let rgb = rife_output.to_rgb8();
    let mut result = image::RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let alpha_a = img_a_rgba.get_pixel(x, y)[3] as f32;
            let alpha_b = img_b_rgba.get_pixel(x, y)[3] as f32;
            let alpha_union = alpha_a.max(alpha_b);

            if alpha_union > 0.0 {
                let alpha_lerp = alpha_a * (1.0 - ratio) + alpha_b * ratio;
                let alpha = alpha_lerp.max(alpha_union * 0.5).min(alpha_union);
                let p = rgb.get_pixel(x, y);
                result.put_pixel(
                    x,
                    y,
                    image::Rgba([p[0], p[1], p[2], alpha.clamp(0.0, 255.0) as u8]),
                );
            }
        }
    }

    DynamicImage::ImageRgba8(result)
}

/// Extract a transparent part from an opaque body-composited RIFE frame.
///
/// RIFE should not receive transparent PNGs directly because fully transparent
/// pixels often have black RGB. This function reverses the body composite so the
/// exported layer can be composited over the same body without dark halos.
pub fn extract_part_from_body_composite(
    rife_output: &DynamicImage,
    body_rgb: &image::RgbImage,
    img_a_rgba: &image::RgbaImage,
    img_b_rgba: &image::RgbaImage,
    ratio: f32,
    width: u32,
    height: u32,
) -> DynamicImage {
    let rgb = rife_output.to_rgb8();
    let mut result = image::RgbaImage::new(width, height);
    let ratio = ratio.clamp(0.0, 1.0);

    for y in 0..height {
        for x in 0..width {
            let alpha_a = img_a_rgba.get_pixel(x, y)[3] as f32;
            let alpha_b = img_b_rgba.get_pixel(x, y)[3] as f32;
            let alpha = alpha_a * (1.0 - ratio) + alpha_b * ratio;
            if alpha < 1.0 {
                continue;
            }

            let a = alpha / 255.0;
            let p = rgb.get_pixel(x, y);
            let b = body_rgb.get_pixel(x, y);
            let unblend = |channel: usize| {
                ((p[channel] as f32 - b[channel] as f32 * (1.0 - a)) / a).clamp(0.0, 255.0) as u8
            };

            result.put_pixel(
                x,
                y,
                image::Rgba([unblend(0), unblend(1), unblend(2), alpha.round() as u8]),
            );
        }
    }

    DynamicImage::ImageRgba8(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_composite_extraction_removes_body_and_keeps_part_color() {
        let mut body = image::RgbImage::new(1, 1);
        body.put_pixel(0, 0, image::Rgb([200, 100, 50]));

        let mut start = image::RgbaImage::new(1, 1);
        start.put_pixel(0, 0, image::Rgba([0, 0, 0, 0]));

        let mut end = image::RgbaImage::new(1, 1);
        end.put_pixel(0, 0, image::Rgba([240, 20, 20, 255]));

        let mut rife = image::RgbImage::new(1, 1);
        rife.put_pixel(0, 0, image::Rgb([220, 60, 35]));

        let extracted = extract_part_from_body_composite(
            &DynamicImage::ImageRgb8(rife),
            &body,
            &start,
            &end,
            0.5,
            1,
            1,
        )
        .to_rgba8();

        let pixel = extracted.get_pixel(0, 0);
        assert!(pixel[0] >= 238);
        assert!(pixel[1] <= 22);
        assert!(pixel[2] <= 22);
        assert_eq!(pixel[3], 128);
    }
}
