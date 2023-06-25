use bfv::{
    parameters::BfvParameters,
    plaintext::{Encoding, Plaintext},
    secret_key::SecretKey,
    Evaluator, PolyType, RelinearizationKey,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use rand::{distributions::Uniform, thread_rng, Rng};
use std::{collections::HashMap, hash::Hash, sync::Arc};

fn bench_bfv(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfv");
    group.sample_size(10);

    for degree in [1 << 15] {
        for m_size in [12, 14, 15] {
            let mut rng = thread_rng();
            let params = BfvParameters::default(m_size, degree);
            let logq = params.poly_ctx(&PolyType::Q, 0).big_q().bits();

            let sk = SecretKey::random(params.degree, &mut rng);

            let mut m0 = params
                .plaintext_modulus_op
                .random_vec(params.degree, &mut rng);
            let mut m1 = params
                .plaintext_modulus_op
                .random_vec(params.degree, &mut rng);

            let evaluator = Evaluator::new(params);

            let pt0 = evaluator.plaintext_encode(&m0, Encoding::default());
            let pt1 = evaluator.plaintext_encode(&m1, Encoding::default());
            let c0 = evaluator.encrypt(&sk, &pt0, &mut rng);
            let c1 = evaluator.encrypt(&sk, &pt1, &mut rng);
            group.bench_function(
                BenchmarkId::new("mul", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter(|| {
                        let _ = evaluator.mul(&c0, &c1);
                    });
                },
            );

            group.bench_function(
                BenchmarkId::new("mul_lazy", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter(|| {
                        let _ = evaluator.mul_lazy(&c0, &c1);
                    });
                },
            );

            let c0_c1 = evaluator.mul(&c0, &c1);
            let rlk_lvl0 = RelinearizationKey::new(evaluator.params(), &sk, 0, &mut rng);
            let mut rlks = HashMap::new();
            rlks.insert(0, rlk_lvl0);
            group.bench_function(
                BenchmarkId::new("relinearize", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter(|| {
                        let _ = evaluator.relinearize(&c0_c1, &rlks);
                    });
                },
            );
        }
    }
}

criterion_group!(bfv, bench_bfv);
criterion_main!(bfv);
