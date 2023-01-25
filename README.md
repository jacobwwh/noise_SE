# An empirical study on noise-handling approaches for SE datasets and models

## Requirements

pytorch

transformers

## Datasets:

Code classification: POJ   https://drive.google.com/file/d/0B2i-vWnOu7MxVlJwQXN6eVNONUU/view?resourcekey=0-Po3kiAifLfCCYnanCBDMHw

Code classification: CodeNet   https://developer.ibm.com/data/project-codenet/

Code search: CodeXGLUE   https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-Adv or https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-WebQuery

Code summarization: CodeXGLUE   https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text

## TODO

|                  |Code classification| Code search|Code summarization (function name)|Code summarization (comment)|
| ---------------- | ------ | -------- | --------- | --------- |
| None      |   |    |     |         |
| Influnce function [1]     |   |    |     |         |
| Meta-weight-net [2]     |   |    |     |         |
| Co-teaching [3]     |   |    |     |         |
| Simifeat [4]     |   |    |     |         |
| Robusttrainer [5]     |   |    |     |         |

### Models

Roberta-based: CodeBERT, (GraphCodeBERT), UniXCoder

Pre-trained encoder-decoder: CodeT5, (PLBART)

Trained-from-scratch models: LSTM, GNN-based approaches

## References

[1] Koh, P. W., & Liang, P. (2017, July). Understanding black-box predictions via influence functions. In International conference on machine learning (pp. 1885-1894). PMLR.

[2] Shu, J., Xie, Q., Yi, L., Zhao, Q., Zhou, S., Xu, Z., & Meng, D. (2019). Meta-weight-net: Learning an explicit mapping for sample weighting. Advances in neural information processing systems, 32.

[3] Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., ... & Sugiyama, M. (2018). Co-teaching: Robust training of deep neural networks with extremely noisy labels. Advances in neural information processing systems, 31.

[4] Zhu, Z., Dong, Z., & Liu, Y. (2022, June). Detecting corrupted labels without training a model to predict. In International Conference on Machine Learning (pp. 27412-27427). PMLR.

[5] Robust Learning of Deep Predictive Models from Noisy and Imbalanced Software Engineering Datasets. In Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering, 2022.

[6] Northcutt, C., Jiang, L., & Chuang, I. (2021). Confident learning: Estimating uncertainty in dataset labels. Journal of Artificial Intelligence Research, 70, 1373-1411.

[7] Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. 2020. Estimating training data influence by tracing gradient descent. Advances in Neural Information Processing Systems 33 (2020), 19920–19930.

