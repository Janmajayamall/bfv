use bfv::*;
use itertools::Itertools;
use rand::{distributions::Uniform, thread_rng, Rng};
use std::sync::Arc;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    let mut rng = thread_rng();
    let params = Arc::new(BfvParameters::new(
        &[60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
        65537,
        1 << 15,
    ));
    let sk = SecretKey::random(&params, &mut rng);

    let mut m1 = rng
        .clone()
        .sample_iter(Uniform::new(0, params.plaintext_modulus))
        .take(params.polynomial_degree)
        .collect_vec();
    let m2 = rng
        .clone()
        .sample_iter(Uniform::new(0, params.plaintext_modulus))
        .take(params.polynomial_degree)
        .collect_vec();
    let pt1 = Plaintext::encode(&m1, &params, Encoding::simd(0));
    let pt2 = Plaintext::encode(&m2, &params, Encoding::simd(0));
    let ct1 = sk.encrypt(&pt1, &mut rng);
    let ct2 = sk.encrypt(&pt2, &mut rng);

    let now = std::time::Instant::now();
    let ct3 = ct1.multiply1(&ct2);
    println!("total time: {:?}", now.elapsed());

    // t com
}
