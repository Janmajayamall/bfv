use crate::parameters::{BfvParameters, PolyType};
use crate::poly::{Poly, Representation};
use crate::warn;
use itertools::{izip, Itertools};
use std::sync::Arc;
use traits::Ntt;

#[derive(Debug, Clone)]
pub struct Ciphertext {
    pub(crate) c: Vec<Poly>,
    pub poly_type: PolyType,
    pub level: usize,
}

// 1. Add
// 2. AddAssign
// 3. Sub
// 4. SubAssign
// 5. MulLazy
// 6. Mul
// 7. Plaintext Add, Sub, Mul
// 8. Relinerization
// 9. Rotations

struct Evaluator<T: Ntt> {
    pub params: BfvParameters<T>,
}

impl<T> Evaluator<T>
where
    T: Ntt,
{
    fn mul(&self, lhs: &mut Ciphertext, rhs: &Ciphertext) -> Ciphertext {
        let mut res = self.mul_lazy(lhs, rhs);
        self.scale_and_round(&mut res);
        res
    }

    fn mul_lazy(&self, lhs: &mut Ciphertext, rhs: &Ciphertext) -> Ciphertext {
        debug_assert!(lhs.c.len() == 2);
        debug_assert!(rhs.c.len() == 2);
        #[cfg(debug_assertions)]
        {
            // We save 2 ntts if polynomial passed to `fast_expand_crt_basis_p_over_q` is in coefficient form. Hence
            // it is cheaper to pass ciphertexts in coefficient form. But if you are stuck with two ciphertext one in coefficient
            // and another in evaluation, pass the one in evaluation form as `self`. This way ciphertext in coefficient
            // form is passed to `fast_expand_crt_basis_p_over_q`  giving us same saving as if both ciphertexts were
            // in coefficient form.
            if (lhs.c[0].representation != rhs.c[0].representation)
                && (rhs.c[0].representation != Representation::Coefficient)
            {
                panic!("Different representation in multiply1 only allows when self is in `Evalaution`")
            }
        }
        assert!(lhs.level == rhs.level);
        assert!(lhs.poly_type == rhs.poly_type);
        assert!(lhs.poly_type == PolyType::Q);

        let level = lhs.level;
        let q_ctx = self.params.poly_ctx(&PolyType::Q, level);
        let p_ctx = self.params.poly_ctx(&PolyType::P, level);
        let pq_ctx = self.params.poly_ctx(&PolyType::PQ, level);

        // let mut now = std::time::Instant::now();
        let mut c00 = q_ctx.expand_crt_basis(
            &lhs.c[0],
            &pq_ctx,
            &p_ctx,
            &self.params.ql_hat_modpl[level],
            &self.params.ql_hat_inv_modql[level],
            &self.params.ql_hat_inv_modql_shoup[level],
            &self.params.ql_inv[level],
            &self.params.alphal_modpl[level],
        );
        let mut c01 = q_ctx.expand_crt_basis(
            &lhs.c[1],
            &pq_ctx,
            &p_ctx,
            &self.params.ql_hat_modpl[level],
            &self.params.ql_hat_inv_modql[level],
            &self.params.ql_hat_inv_modql_shoup[level],
            &self.params.ql_inv[level],
            &self.params.alphal_modpl[level],
        );
        // println!("Extend1 {:?}", now.elapsed());
        if c00.representation != Representation::Evaluation {
            pq_ctx.change_representation(&mut c00, Representation::Evaluation);
            pq_ctx.change_representation(&mut c01, Representation::Evaluation);
        }
        // println!("Extend1 (In Evaluation) {:?}", now.elapsed());

        // now = std::time::Instant::now();
        let mut c10 = q_ctx.fast_expand_crt_basis_p_over_q(
            &rhs.c[0],
            &p_ctx,
            &pq_ctx,
            &self.params.neg_pql_hat_inv_modql[level],
            &self.params.neg_pql_hat_inv_modql_shoup[level],
            &self.params.ql_inv[level],
            &self.params.ql_inv_modpl[level],
            &self.params.pl_hat_modql[level],
            &self.params.pl_hat_inv_modpl[level],
            &self.params.pl_hat_inv_modpl_shoup[level],
            &self.params.pl_inv[level],
            &self.params.alphal_modql[level],
        );
        let mut c11 = q_ctx.fast_expand_crt_basis_p_over_q(
            &rhs.c[1],
            &p_ctx,
            &pq_ctx,
            &self.params.neg_pql_hat_inv_modql[level],
            &self.params.neg_pql_hat_inv_modql_shoup[level],
            &self.params.ql_inv[level],
            &self.params.ql_inv_modpl[level],
            &self.params.pl_hat_modql[level],
            &self.params.pl_hat_inv_modpl[level],
            &self.params.pl_hat_inv_modpl_shoup[level],
            &self.params.pl_inv[level],
            &self.params.alphal_modql[level],
        );
        // println!("Extend2 {:?}", now.elapsed());
        pq_ctx.change_representation(&mut c10, Representation::Evaluation);
        pq_ctx.change_representation(&mut c11, Representation::Evaluation);
        // println!("Extend2 (In Evaluation) {:?}", now.elapsed());

        // now = std::time::Instant::now();
        // tensor
        // c00 * c10
        let c_r0 = pq_ctx.mul(&c00, &c10);

        // c00 * c11 + c01 * c10
        pq_ctx.mul_assign(&mut c00, &c11);
        pq_ctx.mul_assign(&mut c10, &c01);
        pq_ctx.add_assign(&mut c00, &c10);

        // c01 * c11
        pq_ctx.mul_assign(&mut c01, &c11);
        // println!("Tensor {:?}", now.elapsed());

        Ciphertext {
            c: vec![c_r0, c00, c01],
            poly_type: PolyType::PQ,
            level: level,
        }
    }

    fn scale_and_round(&self, c0: &mut Ciphertext) -> Ciphertext {
        // debug_assert!(c0.c[0].representation == Representation::E)
        assert!(c0.poly_type == PolyType::PQ);
        let level = c0.level;
        let pq_ctx = self.params.poly_ctx(&PolyType::PQ, level);
        let q_ctx = self.params.poly_ctx(&PolyType::Q, level);
        let p_ctx = self.params.poly_ctx(&PolyType::P, level);

        let c =
            c0.c.iter_mut()
                .map(|pq_poly| {
                    pq_ctx.change_representation(pq_poly, Representation::Coefficient);
                    pq_ctx.scale_and_round(
                        pq_poly,
                        &q_ctx,
                        &p_ctx,
                        &q_ctx,
                        &self.params.tql_pl_hat_inv_modpl_divpl_modql[level],
                        &self.params.tql_pl_hat_inv_modpl_divpl_frachi[level],
                        &self.params.tql_pl_hat_inv_modpl_divpl_fraclo[level],
                    )
                })
                .collect_vec();

        Ciphertext {
            c,
            poly_type: PolyType::Q,
            level,
        }
    }

    // fn relinearize(&self, c0: Ciphertext) -> Ciphertext {

    // }

    fn add_assign(&self, c0: &mut Ciphertext, c1: &Ciphertext) {
        // TODO: perform checks
        let ctx = self.params.poly_ctx(&c0.poly_type, c0.level);

        izip!(c0.c.iter_mut(), c1.c.iter()).for_each(|(p0, p1)| {
            ctx.add_assign(p0, p1);
        });
    }

    fn add(&self, c0: &Ciphertext, c1: &Ciphertext) -> Ciphertext {
        let ctx = self.params.poly_ctx(&c0.poly_type, c0.level);

        let c = izip!(c0.c.iter(), c1.c.iter())
            .map(|(p0, p1)| ctx.add(p0, p1))
            .collect_vec();

        Ciphertext {
            c,
            poly_type: c0.poly_type.clone(),
            level: c0.level,
        }
    }

    fn sub_assign(&self, c0: &mut Ciphertext, c1: &Ciphertext) {
        // TODO: perform checks
        let ctx = self.params.poly_ctx(&c0.poly_type, c0.level);

        izip!(c0.c.iter_mut(), c1.c.iter()).for_each(|(p0, p1)| {
            ctx.sub_assign(p0, p1);
        });
    }

    fn sub(&self, c0: &Ciphertext, c1: &Ciphertext) -> Ciphertext {
        let ctx = self.params.poly_ctx(&c0.poly_type, c0.level);

        let c = izip!(c0.c.iter(), c1.c.iter())
            .map(|(p0, p1)| ctx.sub(p0, p1))
            .collect_vec();

        Ciphertext {
            c,
            poly_type: c0.poly_type.clone(),
            level: c0.level,
        }
    }
}
