use concrete_ntt::prime64::Plan;

#[derive(Debug, Clone)]
pub struct NttOperator(Plan);

impl traits::Ntt for NttOperator {
    fn new(degree: usize, prime: u64) -> Self {
        let plan = Plan::try_new(degree, prime).unwrap();
        NttOperator(plan)
    }

    fn forward(&self, a: &mut [u64]) {
        self.0.fwd(a);
    }

    fn backward(&self, a: &mut [u64]) {
        self.0.inv(a);
        self.0.normalize(a);
    }

    fn forward_lazy(&self, a: &mut [u64]) {
        self.forward(a);
    }
}

impl PartialEq for NttOperator {
    fn eq(&self, other: &Self) -> bool {
        self.0.modulus() == other.0.modulus() && self.0.ntt_size() == other.0.ntt_size()
    }
}
