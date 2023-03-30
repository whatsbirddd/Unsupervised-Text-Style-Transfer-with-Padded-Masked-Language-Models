# Style Transfer

pytorch implementation   
Author : Saeran Park.  
Unsupervised Text Style Transfer with Padded Masked Language Models (EMNLP 2020)[[Paper]](https://arxiv.org/pdf/2010.01054.pdf)

Stage1. Train the model
---
I trained a single model instead of training two seperate models for source and target style.   
It can be to add a domain embedding to each token emd as proposed by Wu et al.[[Paper]](https://arxiv.org/pdf/1908.08039.pdf)
<pre><code>python train.py </code></pre> 

Stage2. Make a masked token for style transfer
---
<pre><code>python mask.py </code></pre> 

Stage3. Generate tokens
---
You can generate target style text for each sentence on `generate.ipynb`.
