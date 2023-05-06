use std::time::Duration;

use bfv::{
    poly::{Poly, PolyContext, Representation},
    BfvParameters,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::thread_rng;

fn bench_poly(c: &mut Criterion) {
    let mut group = c.benchmark_group("poly");
    group.measurement_time(Duration::new(10, 0));
    group.sample_size(100);

    let plaintext_modulus = 65537;
    let degree = 1 << 15;

    let mut rng = thread_rng();

    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    {
        let params = BfvParameters::new(&[60, 60], plaintext_modulus, degree);
        let q_ctx = params.ciphertext_poly_contexts[1].clone();
        let q = Poly::random(&q_ctx, &Representation::Coefficient, &mut rng);
        group.bench_function(BenchmarkId::new("scale_and_round_decryption", ""), |b| {
            b.iter(|| {
                q.scale_and_round_decryption(
                    &params.plaintext_modulus_op,
                    params.max_bit_size_by2,
                    &params.t_qlhat_inv_modql_divql_modt[1],
                    &params.t_bqlhat_inv_modql_divql_modt[1],
                    &params.t_qlhat_inv_modql_divql_frac[1],
                    &params.t_bqlhat_inv_modql_divql_frac[1],
                )
            })
        });
    }

    {
        let bfv_params = BfvParameters::new(
            &[60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
            plaintext_modulus,
            degree,
        );

        let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
        let p_context = bfv_params.extension_poly_contexts[0].clone();
        let pq_context = bfv_params.pq_poly_contexts[0].clone();

        let pq_poly = Poly::random(&pq_context, &Representation::Coefficient, &mut rng);

        group.bench_function(BenchmarkId::new("scale_and_round", ""), |b| {
            b.iter(|| {
                pq_poly.scale_and_round(
                    &q_context,
                    &p_context,
                    &q_context,
                    &bfv_params.tql_p_hat_inv_modp_divp_modql[0],
                    &bfv_params.tql_p_hat_inv_modp_divp_frac_hi[0],
                    &bfv_params.tql_p_hat_inv_modp_divp_frac_lo[0],
                );
            })
        });
    }

    {
        let bfv_params = BfvParameters::new(
            &[60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
            plaintext_modulus,
            degree,
        );

        let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
        let p_context = bfv_params.extension_poly_contexts[0].clone();
        let mut q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

        group.bench_function(BenchmarkId::new("fast_conv_p_over_q", ""), |b| {
            b.iter(|| {
                q_poly.fast_conv_p_over_q(
                    &p_context,
                    &bfv_params.neg_pql_hat_inv_modql[0],
                    &bfv_params.ql_inv_modp[0],
                );
            })
        });
    }

    {
        let bfv_params = BfvParameters::new(
            &[60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
            plaintext_modulus,
            degree,
        );

        let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
        let p_context = bfv_params.extension_poly_contexts[0].clone();
        let pq_context = bfv_params.pq_poly_contexts[0].clone();
        let mut q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

        group.bench_function(
            BenchmarkId::new("fast_expand_crt_basis_p_over_q", ""),
            |b| {
                b.iter(|| {
                    q_poly.fast_expand_crt_basis_p_over_q(
                        &p_context,
                        &pq_context,
                        &bfv_params.neg_pql_hat_inv_modql[0],
                        &bfv_params.ql_inv_modp[0],
                        &bfv_params.pl_hat_modq[0],
                        &bfv_params.pl_hat_inv_modpl[0],
                        &bfv_params.pl_hat_inv_modpl_shoup[0],
                        &bfv_params.pl_inv[0],
                        &bfv_params.alphal_modq[0],
                    );
                })
            },
        );
    }

    {
        let bfv_params = BfvParameters::new(
            &[60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
            plaintext_modulus,
            degree,
        );

        let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
        let p_context = bfv_params.extension_poly_contexts[0].clone();
        let mut q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

        group.bench_function(BenchmarkId::new("switch_crt_basis", ""), |b| {
            b.iter(|| {
                q_poly.switch_crt_basis(
                    &p_context,
                    &bfv_params.ql_hat_modp[0],
                    &bfv_params.ql_hat_inv_modql[0],
                    &bfv_params.ql_hat_inv_modql_shoup[0],
                    &bfv_params.ql_inv[0],
                    &bfv_params.alphal_modp[0],
                );
            })
        });
    }
}

criterion_group!(poly, bench_poly);
criterion_main!(poly);
