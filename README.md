# Flexible-and-separable-convolution-for-a-better-faster-and-lighter-architecture

## Citation
>>>@article{ZHU2023109589,<br>
title = {FSConv: Flexible and separable convolution for convolutional neural networks compression},<br>
journal = {Pattern Recognition},<br>
volume = {140},<br>
pages = {109589},<br>
year = {2023},<br>
issn = {0031-3203},<br>
doi = {https://doi.org/10.1016/j.patcog.2023.109589},<br>
url = {https://www.sciencedirect.com/science/article/pii/S003132032300290X},<br>
author = {Yangyang Zhu and Luofeng Xie and Zhengfeng Xie and Ming Yin and Guofu Yin},<br>
keywords = {CNNs compression, Representative feature maps, Redundant feature maps, Intrinsic information, Tiny hidden details},<br>
abstract = {Because of limited computation resources, convolutional neural networks (CNNs) are difficult to deploy on mobile devices. To overcome this issue, many methods have successively reduced parameters in CNNs with the idea of removing redundancy among feature maps. We observe similarities between feature maps at the same layer but not complete consistency. Intuitively, the difference between similar feature maps is an essential ingredient for the success of CNNs. Therefore, we propose a flexible and separable convolution (FSConv) in a different perspective to embrace redundancy while requiring less computation, which can implicitly cluster feature maps into different clusters without introducing similarity measurements. Our proposed model extracts intrinsic information from the representative part through ordinary convolution in each cluster and reveals tiny hidden details from the redundant part through groupwise/depthwise convolution. Experimental results demonstrate that FSConv-equipped networks always perform better than previous state-of-the-art CNNs compression algorithms. Code is available at https://github.com/Clarkxielf/FSConv-Flexible-and-Separable-Convolution-for-Convolutional-Neural-Networks-Compression.}<br>
}

