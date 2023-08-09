A work in progress

### Features

To enable serialization and deserilization of types enable `serialization` feature. You also ensure that you have Protoc buffer compiler with version >= 23.4 installed. If not, you can install it from [here](https://grpc.io/docs/protoc-installation/#binary-install).

By default `std` feature is enabled and uses [concrete-ntt](https://github.com/zama-ai/concrete-ntt) as the default NTT backend.

You may enable `nightly` feature to enable `nightly` feature of [concrete-ntt]() that accelartes NTT operation on machines with AVX512 instruction set. Make sure to switch to nightly compiler before enabling `nightly`.

With `hexl-ntt` (only on x86) you can swap out [concrete-ntt]() NTT with NTT backend that uses [hexl](https://github.com/intel/hexl).

With `hexl` (only on x86) you can accelerate modulus operation using [hexl](https://github.com/intel/hexl).

For best performance on x86 either use (1) `hexl-ntt` and `hexl` together OR (2) `nightly` and `hexl` together.

Note that both `concrete-ntt/nightly` and `hexl-ntt` use AVX512 instruction set to accelerate NTT operations and their performance does not differ.
