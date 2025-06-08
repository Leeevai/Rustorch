use std::arch::x86_64::*;

pub struct SimdProcessor {
    pub simd_width: usize,
    pub supports_avx2: bool,
    pub supports_avx512: bool,
}

impl SimdProcessor {
    pub fn new() -> Self {
        let supports_avx2 = is_x86_feature_detected!("avx2");
        let supports_avx512 = is_x86_feature_detected!("avx512f");
        
        let simd_width = if supports_avx512 {
            16 // 512 bits / 32 bits per f32
        } else if supports_avx2 {
            8  // 256 bits / 32 bits per f32
        } else {
            4  // 128 bits / 32 bits per f32 (SSE)
        };

        SimdProcessor {
            simd_width,
            supports_avx2,
            supports_avx512,
        }
    }

    #[inline]
    pub fn add_slice(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if self.supports_avx2 {
            unsafe {
                self.add_avx2(a, b, result);
            }
        } else {
            self.add_scalar(a, b, result);
        }
    }

    #[inline]
    pub fn mul_slice(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if self.supports_avx2 {
            unsafe {
                self.mul_avx2(a, b, result);
            }
        } else {
            self.mul_scalar(a, b, result);
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn add_avx2(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let simd_len = len - (len % 8);

        // Process 8 elements at a time
        for i in (0..simd_len).step_by(8) {
            unsafe {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vr = _mm256_add_ps(va, vb);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vr);
            }
        }

        for i in simd_len..len {
            result[i] = a[i] + b[i];
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mul_avx2(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let simd_len = len - (len % 8);

        // Process 8 elements at a time
        for i in (0..simd_len).step_by(8) {
            unsafe {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vr = _mm256_mul_ps(va, vb);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vr);
            }
        }

        // Handle remaining elements
        for i in simd_len..len {
            result[i] = a[i] * b[i];
        }
    }

    #[inline]
    fn add_scalar(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    #[inline]
    fn mul_scalar(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }
}