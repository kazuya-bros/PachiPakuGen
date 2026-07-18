//! Mask-shaping utilities used by the current See-Through expression pipeline.
//!
//! The module keeps its historical filename for now because callers import it
//! directly. Only deterministic in-process image operations live here.

pub fn adjust_mask(
    mask: &[u8],
    width: u32,
    height: u32,
    dilate_radius: i32,
    blur_radius: i32,
) -> Vec<u8> {
    let radius = dilate_radius.clamp(0, 64);
    let blur = blur_radius.clamp(0, 32);
    let dilated = dilate_mask(mask, width, height, radius);
    if blur <= 0 {
        dilated
    } else {
        let blurred = blur_mask(&dilated, width, height, blur);
        dilated
            .iter()
            .zip(blurred)
            .map(|(hard, soft)| soft.min(*hard))
            .collect()
    }
}

fn dilate_mask(mask: &[u8], width: u32, height: u32, radius: i32) -> Vec<u8> {
    if radius <= 0 {
        return mask.to_vec();
    }
    let mut result = vec![0u8; mask.len()];
    let r2 = radius * radius;
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            if mask[(y as u32 * width + x as u32) as usize] > 128 {
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        if dx * dx + dy * dy <= r2 {
                            let nx = x + dx;
                            let ny = y + dy;
                            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                                result[(ny as u32 * width + nx as u32) as usize] = 255;
                            }
                        }
                    }
                }
            }
        }
    }
    result
}

fn blur_mask(mask: &[u8], width: u32, height: u32, radius: i32) -> Vec<u8> {
    if radius <= 0 {
        return mask.to_vec();
    }
    let mut horizontal = vec![0u8; mask.len()];
    let diameter = radius * 2 + 1;
    for y in 0..height as i32 {
        let mut sum = 0u32;
        for x in -radius..=radius {
            let cx = x.clamp(0, width as i32 - 1);
            sum += mask[(y as u32 * width + cx as u32) as usize] as u32;
        }
        for x in 0..width as i32 {
            horizontal[(y as u32 * width + x as u32) as usize] = (sum / diameter as u32) as u8;
            let remove_x = (x - radius).clamp(0, width as i32 - 1);
            let add_x = (x + radius + 1).clamp(0, width as i32 - 1);
            sum = sum
                .saturating_sub(mask[(y as u32 * width + remove_x as u32) as usize] as u32)
                .saturating_add(mask[(y as u32 * width + add_x as u32) as usize] as u32);
        }
    }

    let mut result = vec![0u8; mask.len()];
    for x in 0..width as i32 {
        let mut sum = 0u32;
        for y in -radius..=radius {
            let cy = y.clamp(0, height as i32 - 1);
            sum += horizontal[(cy as u32 * width + x as u32) as usize] as u32;
        }
        for y in 0..height as i32 {
            result[(y as u32 * width + x as u32) as usize] = (sum / diameter as u32) as u8;
            let remove_y = (y - radius).clamp(0, height as i32 - 1);
            let add_y = (y + radius + 1).clamp(0, height as i32 - 1);
            sum = sum
                .saturating_sub(horizontal[(remove_y as u32 * width + x as u32) as usize] as u32)
                .saturating_add(horizontal[(add_y as u32 * width + x as u32) as usize] as u32);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::adjust_mask;

    #[test]
    fn adjust_mask_without_radius_preserves_input() {
        let input = vec![0, 255, 0, 0];
        assert_eq!(adjust_mask(&input, 2, 2, 0, 0), input);
    }

    #[test]
    fn adjust_mask_dilates_nearby_pixels() {
        let mut input = vec![0; 9];
        input[4] = 255;
        let result = adjust_mask(&input, 3, 3, 1, 0);
        assert_eq!(result[4], 255);
        assert_eq!(result[1], 255);
        assert_eq!(result[3], 255);
        assert_eq!(result[5], 255);
        assert_eq!(result[7], 255);
        assert_eq!(result[0], 0);
    }
}
