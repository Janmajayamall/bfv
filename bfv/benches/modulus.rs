use bfv::Modulus;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use rand::{distributions::Uniform, thread_rng, Rng};

fn bench_modulus(c: &mut Criterion) {
    let mut group = c.benchmark_group("modulus");

    let mut rng = thread_rng();

    for prime in [1125899904679937] {
        let modulus = Modulus::new(prime);
        let logq = 64 - prime.leading_zeros();
        for i in [1 << 15] {
            let mut v = modulus.random_vec(i, &mut rng);
            let mut v2 = modulus.random_vec(i, &mut rng);

            group.bench_function(
                BenchmarkId::new("add_mod_naive_vec", format!("n={i}/logq={logq}")),
                |b| {
                    b.iter(|| modulus.add_mod_naive_vec(&mut v, &v2));
                },
            );

            group.bench_function(
                BenchmarkId::new("add_mod_fast_vec", format!("n={i}/logq={logq}")),
                |b| {
                    b.iter(|| modulus.add_mod_fast_vec(&mut v, &v2));
                },
            );

            group.bench_function(
                BenchmarkId::new("sub_mod_naive_vec", format!("n={i}/logq={logq}")),
                |b| {
                    b.iter(|| modulus.sub_mod_naive_vec(&mut v, &v2));
                },
            );

            group.bench_function(
                BenchmarkId::new("sub_mod_fast_vec", format!("n={i}/logq={logq}")),
                |b| {
                    b.iter(|| modulus.sub_mod_fast_vec(&mut v, &v2));
                },
            );

            group.bench_function(
                BenchmarkId::new("mul_mod_naive_vec", format!("n={i}/logq={logq}")),
                |b| {
                    b.iter(|| modulus.mul_mod_naive_vec(&mut v, &v2));
                },
            );

            group.bench_function(
                BenchmarkId::new("mul_mod_fast_vec", format!("n={i}/logq={logq}")),
                |b| {
                    b.iter(|| modulus.mul_mod_fast_vec(&mut v, &v2));
                },
            );

            let v2_shoup = modulus.compute_shoup_vec(&v2);
            group.bench_function(
                BenchmarkId::new("mul_mod_shoup_vec", format!("n={i}/logq={logq}")),
                |b| {
                    b.iter(|| modulus.mul_mod_shoup_vec(&mut v, &v2, &v2_shoup));
                },
            );

            group.bench_function(
                BenchmarkId::new("reduce_vec", format!("n={i}/logq={logq}")),
                |b| {
                    b.iter(|| modulus.reduce_vec(&mut v));
                },
            );

            group.bench_function(
                BenchmarkId::new("reduce_naive_vec", format!("n={i}/logq={logq}")),
                |b| {
                    b.iter(|| modulus.reduce_naive_vec(&mut v));
                },
            );

            let mut v_u128 = rng
                .clone()
                .sample_iter(Uniform::new(0, u128::MAX))
                .take(i)
                .collect_vec();

            group.bench_function(
                BenchmarkId::new("reduce_naive_u128_vec", format!("n={i}/logq={logq}")),
                |b| {
                    b.iter(|| modulus.reduce_naive_u128_vec(&mut v_u128));
                },
            );

            group.bench_function(
                BenchmarkId::new("barrett_reduction_u128_vec", format!("n={i}/logq={logq}")),
                |b| {
                    b.iter(|| modulus.barret_reduction_u128_vec(&mut v_u128));
                },
            );

            group.bench_function(
                BenchmarkId::new(
                    "barrett_reduction_u128_v2_vec",
                    format!("n={i}/logq={logq}"),
                ),
                |b| {
                    b.iter(|| modulus.barret_reduction_u128_v2_vec(&mut v_u128));
                },
            );

            let prime_30bits = 1073643521;
            let modulus_30bits = Modulus::new(prime_30bits);
            let s_v = modulus_30bits.random_vec(i, &mut rng);
            group.bench_function(
                BenchmarkId::new("switch_modulus", format!("n={i}/logq={logq}")),
                |b| {
                    b.iter_batched(
                        || s_v.clone(),
                        |mut vals| Modulus::switch_modulus(&mut vals, prime_30bits, prime),
                        criterion::BatchSize::PerIteration,
                    )
                },
            );
        }
    }
    // bench reduce
    // bench add
    // bench sub
    // bench others
}

criterion_group!(modulus, bench_modulus);
criterion_main!(modulus);
