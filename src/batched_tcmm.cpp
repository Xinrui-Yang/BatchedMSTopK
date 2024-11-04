#include <torch/extension.h>
#include <pybind11/functional.h>

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "batched_topk.h"

namespace py = pybind11;

std::vector<torch:: Tensor> f_batched_topk(torch::Tensor a, int k) {
    auto c = tcmm_batched_topk(a, k);
    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_batched_topk", &f_batched_topk, "TCMM: Fast Batched Topk");
}
