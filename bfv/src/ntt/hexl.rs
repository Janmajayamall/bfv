#[derive(Debug, Clone)]
pub struct NttOperator(hexl_rs::NttOperator);

impl traits::Ntt for NttOperator {
    fn new(degree: usize, prime: u64) -> Self {
        NttOperator(hexl_rs::NttOperator::new(degree, prime))
    }

    fn forward(&self, a: &mut [u64]) {
        self.0.forward(a);
    }

    fn forward_lazy(&self, a: &mut [u64]) {
        self.0.forward_lazy(a);
    }

    fn backward(&self, a: &mut [u64]) {
        self.0.backward(a);
    }
}

impl PartialEq for NttOperator {
    fn eq(&self, other: &Self) -> bool {
        self.0.modulus() == other.0.modulus() && self.0.ntt_size() == other.0.ntt_size()
    }
}
