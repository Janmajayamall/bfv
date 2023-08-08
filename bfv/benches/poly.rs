use bfv::{
    mod_inverse_biguint_u64, BfvParameters, HybridKeySwitchingKey, PolyContext, PolyType,
    Representation, SecretKey,
};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use itertools::{izip, Itertools};
use ndarray::Array2;
use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
use num_traits::{FromPrimitive, ToPrimitive};
use rand::thread_rng;
use std::time::Duration;
use std::{fmt::format, sync::Arc};
use traits::Ntt;

fn bench_poly(c: &mut Criterion) {
    let mut group = c.benchmark_group("poly");
    group.measurement_time(Duration::new(10, 0));
    group.sample_size(100);
    let batch_size = BatchSize::NumBatches(1000);

    let mut rng = thread_rng();

    for degree in [1 << 15] {
        let params = BfvParameters::default(15, degree);
        let level = 0;

        let q_ctx = params.poly_ctx(&PolyType::Q, level);
        let p_ctx = params.poly_ctx(&PolyType::P, level);
        let pq_ctx = params.poly_ctx(&PolyType::PQ, level);
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);

        let q_poly = q_ctx.random(Representation::Coefficient, &mut rng);

        group.bench_function(
            BenchmarkId::new(
                "scale_and_round_decryption",
                format!("n={degree}/level={level}"),
            ),
            |b| {
                b.iter(|| {
                    let _ = q_ctx.scale_and_round_decryption(
                        &q_poly,
                        &params.plaintext_modulus_op,
                        params.max_bit_size_by2,
                        &params.t_ql_hat_inv_modql_divql_modt[level],
                        &params.t_bql_hat_inv_modql_divql_modt[level],
                        &params.t_ql_hat_inv_modql_divql_frac[level],
                        &params.t_bql_hat_inv_modql_divql_frac[level],
                    );
                })
            },
        );

        let pq_poly = pq_ctx.random(Representation::Coefficient, &mut rng);
        group.bench_function(
            BenchmarkId::new("scale_and_round", format!("n={degree}/level={level}")),
            |b| {
                b.iter(|| {
                    let _ = pq_ctx.scale_and_round(
                        &pq_poly,
                        &q_ctx,
                        &p_ctx,
                        &q_ctx,
                        &params.tql_pl_hat_inv_modpl_divpl_modql[level],
                        &params.tql_pl_hat_inv_modpl_divpl_frachi[level],
                        &params.tql_pl_hat_inv_modpl_divpl_fraclo[level],
                    );
                })
            },
        );

        group.bench_function(
            BenchmarkId::new("fast_conv_p_over_q", format!("n={degree}/level={level}")),
            |b| {
                b.iter(|| {
                    let _ = q_ctx.fast_conv_p_over_q(
                        &q_poly,
                        &p_ctx,
                        &params.neg_pql_hat_inv_modql[level],
                        &params.neg_pql_hat_inv_modql_shoup[level],
                        &params.ql_inv[level],
                        &params.ql_inv_modpl[level],
                    );
                })
            },
        );

        group.bench_function(
            BenchmarkId::new("switch_crt_basis", format!("n={degree}/level={level}")),
            |b| {
                b.iter(|| {
                    let _ = q_ctx.switch_crt_basis(
                        &q_poly,
                        &p_ctx,
                        &params.ql_hat_modpl[level],
                        &params.ql_hat_inv_modql[level],
                        &params.ql_hat_inv_modql_shoup[level],
                        &params.ql_inv[level],
                        &params.alphal_modpl[level],
                    );
                })
            },
        );

        group.bench_function(
            BenchmarkId::new(
                "fast_expand_crt_basis_p_over_q",
                format!("n={degree}/level={level}"),
            ),
            |b| {
                b.iter(|| {
                    let _ = q_ctx.fast_expand_crt_basis_p_over_q(
                        &q_poly,
                        &p_ctx,
                        &pq_ctx,
                        &params.neg_pql_hat_inv_modql[level],
                        &params.neg_pql_hat_inv_modql_shoup[level],
                        &params.ql_inv[level],
                        &params.ql_inv_modpl[level],
                        &params.pl_hat_modql[level],
                        &params.pl_hat_inv_modpl[level],
                        &params.pl_hat_inv_modpl_shoup[level],
                        &params.pl_inv[level],
                        &params.alphal_modql[level],
                    );
                })
            },
        );

        group.bench_function(
            BenchmarkId::new("expand_crt_basis", format!("n={degree}/level={level}")),
            |b| {
                b.iter(|| {
                    let _ = q_ctx.expand_crt_basis(
                        &q_poly,
                        &pq_ctx,
                        &p_ctx,
                        &params.ql_hat_modpl[level],
                        &params.ql_hat_inv_modql[level],
                        &params.ql_hat_inv_modql_shoup[level],
                        &params.ql_inv[level],
                        &params.alphal_modpl[level],
                    );
                })
            },
        );

        group.bench_function(
            BenchmarkId::new("mod_down_next", format!("n={degree}/level={level}")),
            |b| {
                b.iter_batched(
                    || q_poly.clone(),
                    |mut p| {
                        q_ctx.mod_down_next(&mut p, &params.lastq_inv_modql[0]);
                    },
                    batch_size,
                )
            },
        );

        // We need to additonal pre-computes for approx_switch_crt_basis and approx_mod_down. For approx_switch_crt_basis
        // we generate a polynomial in specialP basis (moduli chain of length 3) and switch it to Q basis. This imitates the behaviour
        // of approx_switch_crt_basis in hybrid key switching, where a polynomial with moduli chain of 3 is switched to polynomial with moduli chain of atmost size Q.
        // For approx_mod_down we switch a polynomial in QP basis to Q basis. This again imitates the expected behaviour
        // of approx_mod_down in hybrid key switching.
        {
            let specialp_ctx = params.poly_ctx(&PolyType::SpecialP, level);
            let p = specialp_ctx.big_q();
            let mut p_hat_inv_modp = vec![];
            let mut p_hat_modq = vec![];
            specialp_ctx.iter_moduli_ops().for_each(|(modpi)| {
                p_hat_inv_modp.push(
                    mod_inverse_biguint_u64(&(&p / modpi.modulus()), modpi.modulus())
                        .to_u64()
                        .unwrap(),
                );
            });
            q_ctx.iter_moduli_ops().for_each(|modqj| {
                specialp_ctx.iter_moduli_ops().for_each(|(modpi)| {
                    p_hat_modq.push(((&p / modpi.modulus()) % modqj.modulus()).to_u64().unwrap());
                });
            });
            let p_hat_modq = Array2::from_shape_vec(
                (q_ctx.moduli_count(), specialp_ctx.moduli_count()),
                p_hat_modq,
            )
            .unwrap();

            let mut p_inv_modq = vec![];
            q_ctx.iter_moduli_ops().for_each(|modqi| {
                p_inv_modq.push(
                    mod_inverse_biguint_u64(&p, modqi.modulus())
                        .to_u64()
                        .unwrap(),
                );
            });

            let specialp_poly = specialp_ctx.random(Representation::Coefficient, &mut rng);
            group.bench_function(BenchmarkId::new("approx_switch_crt_basis", ""), |b| {
                b.iter(|| {
                    PolyContext::approx_switch_crt_basis(
                        &specialp_poly.coefficients().view(),
                        specialp_ctx.moduli_ops(),
                        degree,
                        &p_hat_inv_modp,
                        &p_hat_modq,
                        q_ctx.moduli_ops(),
                    );
                })
            });

            let qp_poly = qp_ctx.random(Representation::Evaluation, &mut rng);
            group.bench_function(BenchmarkId::new("approx_mod_down", ""), |b| {
                b.iter_batched(
                    || qp_poly.clone(),
                    |p| {
                        let _ = qp_ctx.approx_mod_down(
                            p,
                            &q_ctx,
                            &specialp_ctx,
                            &p_hat_inv_modp,
                            &p_hat_modq,
                            &p_inv_modq,
                        );
                    },
                    batch_size,
                )
            });
        }
    }
}

criterion_group!(poly, bench_poly);
criterion_main!(poly);
