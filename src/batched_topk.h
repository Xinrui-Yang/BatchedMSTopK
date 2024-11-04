#include <torch/extension.h>

std::vector<torch::Tensor> tcmm_batched_topk(torch::Tensor a, int k);