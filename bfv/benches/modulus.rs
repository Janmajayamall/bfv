use bfv::modulus::Modulus;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use rand::{distributions::Uniform, thread_rng, Rng};

const PRIME: u64 = 1152921504606748673;

fn bench_modulus(c: &mut Criterion) {
    let mut group = c.benchmark_group("modulus");

    let mut rng = thread_rng();
    let modulus = Modulus::new(PRIME);

    for i in [1 << 15] {
        let mut v = modulus.random_vec(i, &mut rng);
        let mut v2 = modulus.random_vec(i, &mut rng);

        group.bench_function(BenchmarkId::new("add_mod_naive_vec", i), |b| {
            b.iter(|| modulus.add_mod_naive_vec(&mut v, &v2));
        });

        group.bench_function(BenchmarkId::new("add_mod_fast_vec", i), |b| {
            b.iter(|| modulus.add_mod_fast_vec(&mut v, &v2));
        });

        group.bench_function(BenchmarkId::new("sub_mod_naive_vec", i), |b| {
            b.iter(|| modulus.sub_mod_naive_vec(&mut v, &v2));
        });

        group.bench_function(BenchmarkId::new("sub_mod_fast_vec", i), |b| {
            b.iter(|| modulus.sub_mod_fast_vec(&mut v, &v2));
        });

        group.bench_function(BenchmarkId::new("mul_mod_naive_vec", i), |b| {
            b.iter(|| modulus.mul_mod_naive_vec(&mut v, &v2));
        });

        group.bench_function(BenchmarkId::new("mul_mod_fast_vec", i), |b| {
            b.iter(|| modulus.mul_mod_fast_vec(&mut v, &v2));
        });

        let v2_shoup = modulus.compute_shoup_vec(&v2);
        group.bench_function(BenchmarkId::new("mul_mod_shoup_vec", i), |b| {
            b.iter(|| modulus.mul_mod_shoup_vec(&mut v, &v2, &v2_shoup));
        });

        group.bench_function(BenchmarkId::new("reduce_vec", i), |b| {
            b.iter(|| modulus.reduce_vec(&mut v));
        });

        group.bench_function(BenchmarkId::new("reduce_naive_vec", i), |b| {
            b.iter(|| modulus.reduce_naive_vec(&mut v));
        });

        let mut v_u128 = rng
            .clone()
            .sample_iter(Uniform::new(0, u128::MAX))
            .take(i)
            .collect_vec();

        group.bench_function(BenchmarkId::new("reduce_naive_u128_vec", i), |b| {
            b.iter(|| modulus.reduce_naive_u128_vec(&mut v_u128));
        });

        group.bench_function(BenchmarkId::new("barrett_reduction_u128_vec", i), |b| {
            b.iter(|| modulus.barret_reduction_u128_vec(&mut v_u128));
        });

        group.bench_function(BenchmarkId::new("barrett_reduction_u128_v2_vec", i), |b| {
            b.iter(|| modulus.barret_reduction_u128_v2_vec(&mut v_u128));
        });
    }

    // bench reduce
    // bench add
    // bench sub
    // bench others
}

criterion_group!(modulus, bench_modulus);
criterion_main!(modulus);
