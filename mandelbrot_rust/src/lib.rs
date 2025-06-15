use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;
use ndarray::Array2;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

#[pyfunction]
fn compute_mandelbrot(
    py: Python,
    width: usize,
    height: usize,
    max_iter: usize,
    re_min: f64,
    re_max: f64,
    im_min: f64,
    im_max: f64,
) -> Py<PyArray2<i32>> {
    // Create the output array
    let mut result = Array2::zeros((height, width));
    
    // Compute the step sizes
    let re_step = (re_max - re_min) / (width as f64);
    let im_step = (im_max - im_min) / (height as f64);
    
    // Parallel computation using rayon
    result.axis_iter_mut(ndarray::Axis(0)).enumerate().par_bridge().for_each(|(y, mut row)| {
        for x in 0..width {
            let c_re = re_min + (x as f64) * re_step;
            let c_im = im_min + (y as f64) * im_step;
            
            let mut z_re = 0.0;
            let mut z_im = 0.0;
            let mut i = 0;
            
            while z_re * z_re + z_im * z_im <= 4.0 && i < max_iter {
                let z_re_sq = z_re * z_re;
                let z_im_sq = z_im * z_im;
                z_im = 2.0 * z_re * z_im + c_im;
                z_re = z_re_sq - z_im_sq + c_re;
                i += 1;
            }
            
            row[x] = i as i32;
        }
    });
    
    result.to_pyarray(py).into()
}

/// A Python module implemented in Rust.
#[pymodule]
fn mandelbrot_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_mandelbrot, m)?)?;
    Ok(())
}
