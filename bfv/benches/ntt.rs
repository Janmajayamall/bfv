use bfv::{Modulus, NttOperator};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::thread_rng;
use traits::Ntt;

fn bench_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt");

    let mut rng = thread_rng();

    for prime in [1125899904679937u64] {
        let logq = 64 - prime.leading_zeros();
        for i in [1 << 15] {
            let ntt = NttOperator::new(i, prime);
            let modulus = Modulus::new(prime);

            let mut v = modulus.random_vec(i, &mut rng);

            group.bench_function(
                BenchmarkId::new("forward", format!("n={i}/logq={logq}")),
                |b| {
                    b.iter_batched(
                        || v.clone(),
                        |mut v0| ntt.forward(&mut v0),
                        BatchSize::PerIteration,
                    );
                },
            );

            group.bench_function(
                BenchmarkId::new("backward", format!("n={i}/logq={logq}")),
                |b| {
                    b.iter_batched(
                        || v.clone(),
                        |mut v0| ntt.backward(&mut v0),
                        BatchSize::PerIteration,
                    );
                },
            );
        }
    }
}

criterion_group!(ntt, bench_ntt);
criterion_main!(ntt);
