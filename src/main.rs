use bfv::*;
use itertools::Itertools;
use rand::{distributions::Uniform, thread_rng, Rng};
use std::sync::Arc;

fn switch_crt_basis() {
    let mut rng = thread_rng();
    let bfv_params = BfvParameters::default(15, 1 << 15);
    let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
    let p_context = bfv_params.extension_poly_contexts[0].clone();
    let mut q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

    for _ in 0..10000 {
        let _ = q_poly.switch_crt_basis(
            &p_context,
            &bfv_params.ql_hat_modp[0],
            &bfv_params.ql_hat_inv_modql[0],
            &bfv_params.ql_hat_inv_modql_shoup[0],
            &bfv_params.ql_inv[0],
            &bfv_params.alphal_modp[0],
        );
    }
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();
    switch_crt_basis();
    // let mut rng = thread_rng();
    // let params = Arc::new(BfvParameters::new(
    //     &[60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
    //     65537,
    //     1 << 15,
    // ));
    // let sk = SecretKey::random(&params, &mut rng);

    // let mut m1 = rng
    //     .clone()
    //     .sample_iter(Uniform::new(0, params.plaintext_modulus))
    //     .take(params.polynomial_degree)
    //     .collect_vec();
    // let m2 = rng
    //     .clone()
    //     .sample_iter(Uniform::new(0, params.plaintext_modulus))
    //     .take(params.polynomial_degree)
    //     .collect_vec();
    // let pt1 = Plaintext::encode(&m1, &params, Encoding::simd(0));
    // let pt2 = Plaintext::encode(&m2, &params, Encoding::simd(0));
    // let ct1 = sk.encrypt(&pt1, &mut rng);
    // let ct2 = sk.encrypt(&pt2, &mut rng);

    // let now = std::time::Instant::now();
    // let ct3 = ct1.multiply1(&ct2);
    // println!("total time: {:?}", now.elapsed());
}
