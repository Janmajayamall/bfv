use bfv::*;
use itertools::{izip, Itertools};
use ndarray::Array2;
use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
use num_traits::{FromPrimitive, ToPrimitive};
use rand::{distributions::Uniform, thread_rng, Rng};
use std::sync::Arc;

fn switch_crt_basis() {
    let mut rng = thread_rng();
    let bfv_params = BfvParameters::default(15, 1 << 15);
    let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
    let p_context = bfv_params.extension_poly_contexts[0].clone();
    let mut q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

    for _ in 0..10000 {
        let _ = q_poly.switch_crt_basis(
            &p_context,
            &bfv_params.ql_hat_modp[0],
            &bfv_params.ql_hat_inv_modql[0],
            &bfv_params.ql_hat_inv_modql_shoup[0],
            &bfv_params.ql_inv[0],
            &bfv_params.alphal_modp[0],
        );
    }
}

fn fast_conv_p_over_q() {
    let mut rng = thread_rng();
    let bfv_params = BfvParameters::default(15, 1 << 15);
    let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
    let p_context = bfv_params.extension_poly_contexts[0].clone();
    let mut q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

    for _ in 0..10000 {
        let _ = q_poly.fast_conv_p_over_q(
            &p_context,
            &bfv_params.neg_pql_hat_inv_modql[0],
            &bfv_params.neg_pql_hat_inv_modql_shoup[0],
            &bfv_params.ql_inv[0],
            &bfv_params.ql_inv_modp[0],
        );
    }
}

fn scale_and_round() {
    let mut rng = thread_rng();
    let bfv_params = BfvParameters::default(15, 1 << 15);

    let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
    let p_context = bfv_params.extension_poly_contexts[0].clone();
    let pq_context = bfv_params.pq_poly_contexts[0].clone();

    let pq_poly = Poly::random(&pq_context, &Representation::Coefficient, &mut rng);

    for _ in 0..10000 {
        let _ = pq_poly.scale_and_round(
            &q_context,
            &p_context,
            &q_context,
            &bfv_params.tql_p_hat_inv_modp_divp_modql[0],
            &bfv_params.tql_p_hat_inv_modp_divp_frac_hi[0],
            &bfv_params.tql_p_hat_inv_modp_divp_frac_lo[0],
        );
    }
}

fn ciphertext_mul() {
    let mut rng = thread_rng();
    let params = Arc::new(BfvParameters::default(15, 1 << 15));
    let sk = SecretKey::random(&params, &mut rng);

    let m1 = rng
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

    for _ in 0..1000 {
        let _ = ct1.multiply1(&ct2);
    }
}

fn rotations() {
    let mut rng = thread_rng();
    let params = Arc::new(BfvParameters::default(15, 1 << 15));
    let sk = SecretKey::random(&params, &mut rng);
    let m = rng
        .clone()
        .sample_iter(Uniform::new(0, params.plaintext_modulus))
        .take(params.polynomial_degree)
        .collect_vec();
    let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
    let ct = sk.encrypt(&pt, &mut rng);

    let rot_key = GaloisKey::new(3, &ct.c_ref()[0].context, &sk, &mut rng);

    for _ in 0..1000 {
        let _ = rot_key.rotate(&ct);
    }
}

fn key_switch() {
    let mut rng = thread_rng();
    let params = Arc::new(BfvParameters::default(15, 1 << 15));
    let sk = SecretKey::random(&params, &mut rng);
    let p0 = Poly::random(
        &params.ciphertext_ctx_at_level(0),
        &Representation::Evaluation,
        &mut rng,
    );
    let p1 = Poly::random(
        &params.ciphertext_ctx_at_level(0),
        &Representation::Coefficient,
        &mut rng,
    );
    let ksk = HybridKeySwitchingKey::new(&p0, &sk, &params.ciphertext_ctx_at_level(0), &mut rng);

    for _ in 0..1000 {
        let _ = ksk.switch(&p1);
    }
}

fn approx_switch_crt_basis() {
    let mut rng = thread_rng();
    let degree = 1 << 15;
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

        izip!(p_moduli.iter()).for_each(|pj| q_hat_modp.push(((&q / qi) % pj).to_u64().unwrap()));
    });
    let q_hat_modp =
        Array2::<u64>::from_shape_vec((q_context.moduli.len(), p_moduli.len()), q_hat_modp)
            .unwrap();
    let q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

    for _ in 0..10000 {
        let _ = Poly::approx_switch_crt_basis(
            q_poly.coefficients.view(),
            &q_context.moduli_ops,
            q_context.degree,
            &q_hat_inv_modq,
            &q_hat_modp,
            &p_context.moduli_ops,
        );
    }
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();
    // switch_crt_basis();
    // scale_and_round();
    // fast_conv_p_over_q();
    // ciphertext_mul();
    // key_switch();
    // rotations();
    approx_switch_crt_basis()
}
