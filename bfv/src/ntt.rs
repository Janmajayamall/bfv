use concrete_ntt::prime64::Plan;
use traits::Ntt;

#[derive(Debug, Clone)]
pub struct NttOperator {
    degree: usize,
    prime: u64,
    plan: Plan,
}

impl Ntt for NttOperator {
    // use `concrete-ntt` as native ntt operator
    fn new(degree: usize, prime: u64) -> Self {
        let plan = Plan::try_new(degree, prime).unwrap();
        NttOperator {
            degree,
            prime,
            plan,
        }
    }

    fn forward(&self, a: &mut [u64]) {
        self.plan.fwd(a);
    }

    fn backward(&self, a: &mut [u64]) {
        self.plan.inv(a);
        self.plan.normalize(a);
    }

    fn forward_lazy(&self, a: &mut [u64]) {
        self.forward(a);
    }
}

impl PartialEq for NttOperator {
    fn eq(&self, other: &Self) -> bool {
        self.prime == other.prime && self.degree == other.degree
    }
}
