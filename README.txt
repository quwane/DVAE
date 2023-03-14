train_vae_v1: 是我写的主程序，代码不够简洁，可读性差，你们可以尝试用tensorboard 或者 args 将代码写的可读性更好
diffraction_vae: vae 的前端衍射神经网络，具体的衍射过程是用角谱法实现的，在utils.py 里面，具体的transfer_kernel， diffraction 方法可以通过附带的这篇
OE 文章中找到答案。
（注：vae 的 sigma 和 mu 利用两个不同的D2NN 获得）
同时附带了一个利用相同方法实现角谱模拟的matlab 程序，可供参考。
首先看懂这篇文章，搞清楚matlab和python编程的逻辑，有余力可以改写代码，这个代码具体的思路可能需要我给你们细说一下。
OE: 2009 Matsushima, K. & Shimobaba, T. Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields. Optics Express 17, 19662-19673 (2009). https://doi.org:10.1364/oe.17.019662