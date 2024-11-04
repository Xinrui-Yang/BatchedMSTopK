#include <torch/extension.h>
#include <pybind11/functional.h>

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "topk.h"

namespace py = pybind11;

std::vector<torch:: Tensor> f_topk(torch::Tensor a, int k) {
    auto c = tcmm_topk(a, k);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_topk", &f_topk, "TCMM: Fast Topk");
}
