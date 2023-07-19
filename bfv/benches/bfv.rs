use bfv::{
    BfvParameters, Encoding, EvaluationKey, Evaluator, Plaintext, PolyType, RelinearizationKey,
    Representation, SecretKey,
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

            let sk = SecretKey::random(params.degree, params.hw, &mut rng);

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

            let mut c0_c1_in_pq = evaluator.mul_lazy(&c0, &c1);
            let c0_c1_in_pq_2 = evaluator.mul_lazy(&c0, &c1);
            group.bench_function(
                BenchmarkId::new("add_assign_lazy", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter(|| {
                        let _ = evaluator.add_assign(&mut c0_c1_in_pq, &c0_c1_in_pq_2);
                    });
                },
            );

            let c0_c1 = evaluator.mul(&c0, &c1);
            let ek = EvaluationKey::new(evaluator.params(), &sk, &[0], &[0], &[0], &mut rng);
            group.bench_function(
                BenchmarkId::new("relinearize", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter(|| {
                        let _ = evaluator.relinearize(&c0_c1, &ek);
                    });
                },
            );

            let ek = EvaluationKey::new(evaluator.params(), &sk, &[0], &[0], &[1], &mut rng);
            group.bench_function(
                BenchmarkId::new("rotate", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter(|| {
                        let _ = evaluator.rotate(&c0, 1, &ek);
                    });
                },
            );

            let mut c0_clone = c0.clone();
            group.bench_function(
                BenchmarkId::new("add_assign", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter(|| {
                        let _ = evaluator.add_assign(&mut c0_clone, &c1);
                    });
                },
            );
        }
    }
}

criterion_group!(bfv, bench_bfv);
criterion_main!(bfv);
