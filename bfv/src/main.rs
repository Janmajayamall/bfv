use bfv::{rot_to_galois_element, *};
use itertools::{izip, Itertools};
use ndarray::Array2;
use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
use num_traits::{FromPrimitive, ToPrimitive};
use rand::{distributions::Uniform, thread_rng, Rng};
use std::{collections::HashMap, sync::Arc};

fn switch_crt_basis() {
    let mut rng = thread_rng();
    let params = BfvParameters::default(15, 1 << 15);

    let q_context = params.poly_ctx(&PolyType::Q, 0);
    let p_context = params.poly_ctx(&PolyType::P, 0);
    let mut q_poly = q_context.random(Representation::Coefficient, &mut rng);

    for _ in 0..10000 {
        let _ = q_context.switch_crt_basis(
            &q_poly,
            &p_context,
            &params.ql_hat_modpl[0],
            &params.ql_hat_inv_modql[0],
            &params.ql_hat_inv_modql_shoup[0],
            &params.ql_inv[0],
            &params.alphal_modpl[0],
        );
    }
}

fn scale_and_round() {
    let mut rng = thread_rng();
    let params = BfvParameters::default(15, 1 << 15);

    let q_context = params.poly_ctx(&PolyType::Q, 0);
    let p_context = params.poly_ctx(&PolyType::P, 0);
    let pq_context = params.poly_ctx(&PolyType::PQ, 0);

    let pq_poly = pq_context.random(Representation::Coefficient, &mut rng);

    for _ in 0..10000 {
        let _ = pq_context.scale_and_round(
            &pq_poly,
            &q_context,
            &p_context,
            &q_context,
            &params.tql_pl_hat_inv_modpl_divpl_modql[0],
            &params.tql_pl_hat_inv_modpl_divpl_frachi[0],
            &params.tql_pl_hat_inv_modpl_divpl_fraclo[0],
        );
    }
}

// fn fast_conv_p_over_q() {
//     let mut rng = thread_rng();
//     let bfv_params = BfvParameters::default(15, 1 << 15);
//     let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
//     let p_context = bfv_params.extension_poly_contexts[0].clone();
//     let mut q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

//     for _ in 0..10000 {
//         let _ = q_poly.fast_conv_p_over_q(
//             &p_context,
//             &bfv_params.neg_pql_hat_inv_modql[0],
//             &bfv_params.neg_pql_hat_inv_modql_shoup[0],
//             &bfv_params.ql_inv[0],
//             &bfv_params.ql_inv_modp[0],
//         );
//     }
// }

// fn scale_and_round() {
//     let mut rng = thread_rng();
//     let bfv_params = BfvParameters::default(15, 1 << 15);

//     let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
//     let p_context = bfv_params.extension_poly_contexts[0].clone();
//     let pq_context = bfv_params.pq_poly_contexts[0].clone();

//     let pq_poly = Poly::random(&pq_context, &Representation::Coefficient, &mut rng);

//     for _ in 0..10000 {
//         let _ = pq_poly.scale_and_round(
//             &q_context,
//             &p_context,
//             &q_context,
//             &bfv_params.tql_p_hat_inv_modp_divp_modql[0],
//             &bfv_params.tql_p_hat_inv_modp_divp_frac_hi[0],
//             &bfv_params.tql_p_hat_inv_modp_divp_frac_lo[0],
//         );
//     }
// }

fn ciphertext_mul() {
    let mut rng = thread_rng();
    let params = BfvParameters::default(15, 1 << 15);

    // gen keys
    let sk = SecretKey::random(params.degree, params.hw, &mut rng);
    let ek = EvaluationKey::new(&params, &sk, &[0], &[], &[], &mut rng);

    let mut m0 = params
        .plaintext_modulus_op
        .random_vec(params.degree, &mut rng);
    let m1 = params
        .plaintext_modulus_op
        .random_vec(params.degree, &mut rng);

    let evaluator = Evaluator::new(params);
    let pt0 = evaluator.plaintext_encode(&m0, Encoding::default());
    let pt1 = evaluator.plaintext_encode(&m1, Encoding::default());
    let ct0 = evaluator.encrypt(&sk, &pt0, &mut rng);
    let ct1 = evaluator.encrypt(&sk, &pt1, &mut rng);
    let ct0_ct1 = evaluator.mul(&ct0, &ct1);

    // ciphertext mul
    // for _ in 0..1000 {
    //     evaluator.mul(&ct0, &ct1);
    // }

    // relin
    for _ in 0..1000 {
        evaluator.relinearize(&ct0_ct1, &ek);
    }
}

fn ciphertext_add() {
    let mut rng = thread_rng();
    let params = BfvParameters::default(15, 1 << 15);
    let sk = SecretKey::random(params.degree, params.hw, &mut rng);

    let mut m0 = params
        .plaintext_modulus_op
        .random_vec(params.degree, &mut rng);
    let m1 = params
        .plaintext_modulus_op
        .random_vec(params.degree, &mut rng);

    let evaluator = Evaluator::new(params);
    let pt0 = evaluator.plaintext_encode(&m0, Encoding::default());
    let pt1 = evaluator.plaintext_encode(&m1, Encoding::default());
    let ct0 = evaluator.encrypt(&sk, &pt0, &mut rng);
    let ct1 = evaluator.encrypt(&sk, &pt1, &mut rng);

    for _ in 0..1000 {
        let _ = evaluator.add(&ct0, &ct1);
    }
}

fn ciphertext_rotate() {
    let mut rng = thread_rng();
    let params = BfvParameters::default(15, 1 << 15);

    // gen keys
    let sk = SecretKey::random(params.degree, params.hw, &mut rng);
    let ek = EvaluationKey::new(&params, &sk, &[], &[0], &[0], &mut rng);

    let m0 = params
        .plaintext_modulus_op
        .random_vec(params.degree, &mut rng);

    let evaluator = Evaluator::new(params);
    let pt0 = evaluator.plaintext_encode(&m0, Encoding::default());
    let ct0 = evaluator.encrypt(&sk, &pt0, &mut rng);

    for _ in 0..1000 {
        let _ = evaluator.rotate(&ct0, 1, &ek);
    }
}

// fn rotations() {
//     let mut rng = thread_rng();
//     let params = Arc::new(BfvParameters::default(15, 1 << 15));
//     let sk = SecretKey::random(&params, &mut rng);
//     let m = rng
//         .clone()
//         .sample_iter(Uniform::new(0, params.plaintext_modulus))
//         .take(params.polynomial_degree)
//         .collect_vec();
//     let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
//     let ct = sk.encrypt(&pt, &mut rng);

//     let rot_key = GaloisKey::new(3, &ct.c_ref()[0].context, &sk, &mut rng);

//     for _ in 0..1000 {
//         let _ = rot_key.rotate(&ct);
//     }
// }

// fn key_switch() {
//     let mut rng = thread_rng();
//     let params = Arc::new(BfvParameters::default(15, 1 << 15));
//     let sk = SecretKey::random(&params, &mut rng);
//     let p0 = Poly::random(
//         &params.ciphertext_ctx_at_level(0),
//         &Representation::Evaluation,
//         &mut rng,
//     );
//     let p1 = Poly::random(
//         &params.ciphertext_ctx_at_level(0),
//         &Representation::Coefficient,
//         &mut rng,
//     );
//     let ksk = HybridKeySwitchingKey::new(&p0, &sk, &params.ciphertext_ctx_at_level(0), &mut rng);

//     for _ in 0..1000 {
//         let _ = ksk.switch(&p1);
//     }
// }

// fn approx_switch_crt_basis() {
//     let mut rng = thread_rng();
//     let degree = 1 << 15;
//     let p_moduli = generate_primes_vec(
//         &vec![60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
//         degree,
//         &[],
//     );
//     let q_moduli = generate_primes_vec(&vec![60, 60, 60], degree, &p_moduli);

//     let q_context = Arc::new(PolyContext::<NttOperator>::new(&q_moduli, degree));
//     let p_context = Arc::new(PolyContext::<NttOperator>::new(&p_moduli, degree));

//     // Pre-computation
//     let mut q_hat_inv_modq = vec![];
//     let mut q_hat_modp = vec![];
//     let q = q_context.modulus();
//     let q_dig = q_context.modulus_dig();
//     izip!(q_context.moduli.iter()).for_each(|(qi)| {
//         let qi_hat_inv_modqi = (&q_dig / *qi)
//             .mod_inverse(BigUintDig::from_u64(*qi).unwrap())
//             .unwrap()
//             .to_biguint()
//             .unwrap()
//             .to_u64()
//             .unwrap();

//         q_hat_inv_modq.push(qi_hat_inv_modqi);

//         izip!(p_moduli.iter()).for_each(|pj| q_hat_modp.push(((&q / qi) % pj).to_u64().unwrap()));
//     });
//     let q_hat_modp =
//         Array2::<u64>::from_shape_vec((q_context.moduli.len(), p_moduli.len()), q_hat_modp)
//             .unwrap();
//     let q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

//     for _ in 0..10000 {
//         let _ = Poly::<NttOperator>::approx_switch_crt_basis(
//             &q_poly.coefficients.view(),
//             &q_context.moduli_ops,
//             q_context.degree,
//             &q_hat_inv_modq,
//             &q_hat_modp,
//             &p_context.moduli_ops,
//         );
//     }
// }

// fn approx_mod_down() {
//     let degree = 1 << 15;
//     let q_moduli = generate_primes_vec(
//         &vec![60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
//         degree,
//         &[],
//     );
//     let p_moduli = generate_primes_vec(&vec![60, 60, 60], degree, &q_moduli);
//     let qp_moduli = [q_moduli.clone(), p_moduli.clone()].concat();

//     let q_context = Arc::new(PolyContext::<NttOperator>::new(&q_moduli, degree));
//     let p_context = Arc::new(PolyContext::<NttOperator>::new(&p_moduli, degree));
//     let qp_context = Arc::new(PolyContext::<NttOperator>::new(&qp_moduli, degree));

//     // just few checks
//     let q_size = q_context.moduli.len();
//     let p_size = p_context.moduli.len();
//     let qp_size: usize = q_size + p_size;

//     // Pre computation
//     let p = p_context.modulus();
//     let p_dig = p_context.modulus_dig();
//     let mut p_hat_inv_modp = vec![];
//     let mut p_hat_modq = vec![];
//     p_context.moduli.iter().for_each(|(pi)| {
//         p_hat_inv_modp.push(
//             (&p_dig / pi)
//                 .mod_inverse(BigUintDig::from_u64(*pi).unwrap())
//                 .unwrap()
//                 .to_biguint()
//                 .unwrap()
//                 .to_u64()
//                 .unwrap(),
//         );

//         // pi_hat_modq
//         let p_hat = &p / pi;
//         q_context
//             .moduli
//             .iter()
//             .for_each(|qi| p_hat_modq.push((&p_hat % qi).to_u64().unwrap()));
//     });
//     let p_hat_modq =
//         Array2::from_shape_vec((p_context.moduli.len(), q_context.moduli.len()), p_hat_modq)
//             .unwrap();
//     let mut p_inv_modq = vec![];
//     q_context.moduli.iter().for_each(|qi| {
//         p_inv_modq.push(
//             p_dig
//                 .clone()
//                 .mod_inverse(BigUintDig::from_u64(*qi).unwrap())
//                 .unwrap()
//                 .to_biguint()
//                 .unwrap()
//                 .to_u64()
//                 .unwrap(),
//         );
//     });
//     let mut rng = thread_rng();
//     for _ in 0..10000 {
//         let mut qp_poly = Poly::random(&qp_context, &Representation::Evaluation, &mut rng);
//         let _ = qp_poly.approx_mod_down(
//             &q_context,
//             &p_context,
//             &p_hat_inv_modp,
//             &p_hat_modq,
//             &p_inv_modq,
//         );
//     }
// }

// fn mod_down_next() {
//     let mut rng = thread_rng();
//     let params = BfvParameters::default(15, 1 << 15);
//     let q_ctx = params.ciphertext_ctx_at_level(0);
//     let q_poly = Poly::random(&q_ctx, &Representation::Coefficient, &mut rng);
//     for _ in 0..10000 {
//         let mut p = q_poly.clone();
//         p.mod_down_next(
//             &params.lastq_inv_modq[0],
//             &params.ciphertext_ctx_at_level(1),
//         );
//     }
// }

// fn barrett_reduction_u128() {
//     let modulus = Modulus::new(1152921504606748673);
//     let a: u128 = thread_rng().gen();

//     for _ in 0..100000000 {
//         modulus.barret_reduction_u128(a);
//     }
// }

// fn poly_mul_assign() {
//     let mut rng = thread_rng();
//     let params = BfvParameters::default(4, 1 << 15);
//     let q_ctx = params.ciphertext_ctx_at_level(0);
//     let mut a = Poly::random(&q_ctx, &Representation::Evaluation, &mut rng);
//     let b = Poly::random(&q_ctx, &Representation::Evaluation, &mut rng);

//     for _ in 0..2000 {
//         let _ = &a * &b;
//     }

//     let now = std::time::Instant::now();
//     for _ in 0..100000 {
//         a *= &b;
//     }
//     println!("Time total: {:?}", now.elapsed() / 100000);
// }

// fn mul_mod() {
//     let mut rng = thread_rng();
//     let modulus = Modulus::new(1152921504606748673);
//     let mut a = modulus.random_vec(1 << 15, &mut rng);
//     let b = modulus.random_vec(1 << 15, &mut rng);
//     for _ in 0..1000000 {
//         modulus.mul_mod_fast_vec(&mut a, &b);
//     }
// }

fn main() {
    // switch_crt_basis();
    // scale_and_round();
    // fast_conv_p_over_q();
    // ciphertext_mul();
    // ciphertext_add();
    ciphertext_rotate();
    // key_switch();
    // rotations();
    // approx_switch_crt_basis()
    // approx_mod_down();
    // barrett_reduction_u128();
    // mod_down_next();
    // mul_mod();
    // poly_mul_assign();
}
