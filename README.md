# ML-project
This is the repository for ML project, face verification.

## Pipeline

Face alignment [1] (locating facial landmarks like eyes, nose or mouth)

↓

High dimensional features (based on local binary patterns (LBP))

↓

Mapping to low dimensional features [2] (sparse)

↓

GaussianFace [3]

References:

[1]. X. Cao, Y. Wei, F. Wen, and J. Sun. Face alignment by explicit shape regression. In Computer Vision and Pattern Recognition, pages 2887 –2894, June 2012. 1, 2

[2]. Chen, D.; Cao, X.; Wen, F.; and Sun, J. 2013. Blessing of dimensionality: High-dimensional feature and its efficient compression for face verification. In CVPR.

[3]. Lu, Chaochao, and Xiaoou Tang. "Surpassing human-level face verification performance on LFW with GaussianFace." Twenty-ninth AAAI conference on artificial intelligence. 2015.
