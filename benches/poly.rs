use std::time::Duration;

use bfv::{
    nb_theory::generate_primes_vec,
    poly::{Poly, PolyContext, Representation},
    BfvParameters,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::izip;
use ndarray::Array2;
use num_bigint::BigUint;
use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
use num_traits::{FromPrimitive, ToPrimitive};
use rand::thread_rng;
use std::sync::Arc;

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
                    &bfv_params.neg_pql_hat_inv_modql_shoup[0],
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
                        &bfv_params.neg_pql_hat_inv_modql_shoup[0],
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

    {
        let p_moduli = generate_primes_vec(
            &vec![60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
            degree,
            &[],
        );
        let q_moduli = generate_primes_vec(&vec![60, 60, 60], degree, &p_moduli);

        let q_context = Arc::new(PolyContext::new(&q_moduli, degree));
        let p_context = Arc::new(PolyContext::new(&p_moduli, degree));

        // Pre-computation
        let mut q_hat_inv_modq = vec![];
        let mut q_hat_modp = vec![];
        let q = q_context.modulus();
        let q_dig = q_context.modulus_dig();
        izip!(q_context.moduli.iter()).for_each(|(qi)| {
            let qi_hat_inv_modqi = (&q_dig / *qi)
                .mod_inverse(BigUintDig::from_u64(*qi).unwrap())
                .unwrap()
                .to_biguint()
                .unwrap()
                .to_u64()
                .unwrap();

            q_hat_inv_modq.push(qi_hat_inv_modqi);

            izip!(p_moduli.iter())
                .for_each(|pj| q_hat_modp.push(((&q / qi) % pj).to_u64().unwrap()));
        });
        let q_hat_modp =
            Array2::<u64>::from_shape_vec((q_context.moduli.len(), p_moduli.len()), q_hat_modp)
                .unwrap();
        let q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

        group.bench_function(BenchmarkId::new("approximate_switch_crt_basis", ""), |b| {
            b.iter(|| {
                Poly::approx_switch_crt_basis(
                    &q_poly.coefficients,
                    &q_context.moduli_ops,
                    q_context.degree,
                    &q_hat_inv_modq,
                    &q_hat_modp,
                    &p_context.moduli_ops,
                );
            })
        });
    }

    {
        let q_moduli = generate_primes_vec(
            &vec![60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
            degree,
            &[],
        );
        let p_moduli = generate_primes_vec(&vec![60, 60, 60], degree, &q_moduli);
        let qp_moduli = [q_moduli.clone(), p_moduli.clone()].concat();

        let q_context = Arc::new(PolyContext::new(&q_moduli, degree));
        let p_context = Arc::new(PolyContext::new(&p_moduli, degree));
        let qp_context = Arc::new(PolyContext::new(&qp_moduli, degree));

        // just few checks
        let q_size = q_context.moduli.len();
        let p_size = p_context.moduli.len();
        let qp_size = q_size + p_size;
        izip!(
            qp_context.moduli_ops.iter().skip(q_size),
            p_context.moduli_ops.iter()
        )
        .for_each(|(a, b)| {
            assert_eq!(a, b);
        });
        izip!(
            qp_context.ntt_ops.iter().skip(q_size),
            p_context.ntt_ops.iter()
        )
        .for_each(|(a, b)| {
            assert_eq!(a, b);
        });

        // Pre computation
        let p = p_context.modulus();
        let p_dig = p_context.modulus_dig();
        let mut p_hat_inv_modp = vec![];
        let mut p_hat_modq = vec![];
        p_context.moduli.iter().for_each(|(pi)| {
            p_hat_inv_modp.push(
                (&p_dig / pi)
                    .mod_inverse(BigUintDig::from_u64(*pi).unwrap())
                    .unwrap()
                    .to_biguint()
                    .unwrap()
                    .to_u64()
                    .unwrap(),
            );

            // pi_hat_modq
            let p_hat = &p / pi;
            q_context
                .moduli
                .iter()
                .for_each(|qi| p_hat_modq.push((&p_hat % qi).to_u64().unwrap()));
        });
        let p_hat_modq =
            Array2::from_shape_vec((p_context.moduli.len(), q_context.moduli.len()), p_hat_modq)
                .unwrap();
        let mut p_inv_modq = vec![];
        q_context.moduli.iter().for_each(|qi| {
            p_inv_modq.push(
                p_dig
                    .clone()
                    .mod_inverse(BigUintDig::from_u64(*qi).unwrap())
                    .unwrap()
                    .to_biguint()
                    .unwrap()
                    .to_u64()
                    .unwrap(),
            );
        });

        let mut qp_poly = Poly::random(&qp_context, &Representation::Evaluation, &mut rng);

        group.bench_function(BenchmarkId::new("approximate_mod_down", ""), |b| {
            b.iter(|| {
                Poly::approx_mod_down(
                    &qp_poly.coefficients,
                    &qp_context,
                    &p_context,
                    &p_hat_inv_modp,
                    &p_hat_modq,
                    &p_inv_modq,
                );
            })
        });
    }
}

criterion_group!(poly, bench_poly);
criterion_main!(poly);
