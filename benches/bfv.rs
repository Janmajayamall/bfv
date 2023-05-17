use bfv::{
    parameters::BfvParameters,
    plaintext::{Encoding, Plaintext},
    secret_key::SecretKey,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use pprof::criterion::{Output, PProfProfiler};
use rand::{distributions::Uniform, thread_rng, Rng};
use std::sync::Arc;

fn bench_bfv(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfv");
    group.sample_size(10);

    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    for degree in [1 << 15] {
        for m_size in [14, 15] {
            let mut rng = thread_rng();
            let bfv_params = Arc::new(BfvParameters::default(m_size, degree));
            let logq = bfv_params.ciphertext_poly_contexts[0].modulus().bits();

            let sk = SecretKey::random(&bfv_params, &mut rng);

            let m = rng
                .clone()
                .sample_iter(Uniform::new(0, bfv_params.plaintext_modulus))
                .take(bfv_params.polynomial_degree)
                .collect_vec();
            let pt = Plaintext::encode(&m, &bfv_params, Encoding::simd(0));
            let ct = sk.encrypt(&pt, &mut rng);
            let ct2 = sk.encrypt(&pt, &mut rng);

            group.bench_function(
                BenchmarkId::new("multiplication1", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter(|| {
                        ct.multiply1(&ct2);
                    });
                },
            );
        }
    }
}

criterion_group!(
    name = bfv;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_bfv
);
criterion_main!(bfv);
