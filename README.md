<img src="./block-recurrent-transformer.png" width="450px"></img>

## Block Recurrent Transformer - Pytorch

Implementation of <a href="https://arxiv.org/abs/2203.07852">Block Recurrent Transformer</a> - Pytorch. The highlight of the paper is its reported ability to remember something up to 60k tokens ago.

## Appreciation

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work and open source cutting edge artificial intelligence research

## Install

```bash
$ pip install block-recurrent-transformer
```

## Usage

```python
import torch
from block_recurrent_transformer_pytorch import BlockRecurrentTransformer

model = BlockRecurrentTransformer(
    num_tokens = 20000,             # vocab size
    dim = 512,                      # model dimensions
    depth = 6,                      # depth
    dim_head = 64,                  # attention head dimensions
    heads = 8,                      # number of attention heads
    max_seq_len = 1024,             # the total receptive field of the transformer, in the paper this was 2 * block size
    block_width = 512,              # block size - total receptive field is max_seq_len, 2 * block size in paper. the block furthest forwards becomes the new cached xl memories, which is a block size of 1 (please open an issue if i am wrong)
    xl_memories_layers = (5, 6),    # which layers to use xl memories. very old deepmind papers have shown you only need the last penultimate layers to have cached key values to see majority of benefit
    num_state_vectors = 512,        # number of state vectors, i believe this was a single block size in the paper, but can be any amount
    recurrent_layers = (4,),        # where to place the recurrent layer(s) for states with fixed simple gating
    enhanced_recurrence = True      # enhanced recurrence from ernie-doc paper, i have seen it to work well on my local machine
)

seq = torch.randint(0, 2000, (1, 1024))

out, mems1, states1 = model(seq)
out, mems2, states2 = model(seq, xl_memories = mems1, states = states1)
out, mems3, states3 = model(seq, xl_memories = mems2, states = states2)
```

## Test on Enwik8

First `pip install -r requirements.txt`, then

```bash
$ python train.py
```

## Todo

- [x] use dynamic positional bias
- [x] add enhanced recurrence
- [x] setup local attention blocks, as in the paper
- [x] wrapper transformer class for training
- [x] take care of generation with recurrence in `RecurrentTrainWrapper`
- [x] add ability to dropout to entire memories and states during each segment step during trainng
- [x] test full system on enwik8 locally and ablate states and memories and see effects first  hand

- [ ] think about giving memories information to the dynamic pos bias mlp
- [ ] run a few experiments of fixed gating in regular transformers
- [ ] make sure attention class can support batch-less dimensions (conditionals on einsum equations) and simplify some logic - allow for single head key / values too
- [ ] revisit <a href="https://github.com/lucidrains/memformer">memformer</a>

## Citations

```bibtex
@article{Hutchins2022BlockRecurrentT,
    title   = {Block-Recurrent Transformers},
    author  = {DeLesley S. Hutchins and Imanol Schlag and Yuhuai Wu and Ethan Dyer and Behnam Neyshabur},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2203.07852}
}
```

```bibtex
@article{Ding2021ERNIEDocAR,
    title   = {ERNIE-Doc: A Retrospective Long-Document Modeling Transformer},
    author  = {Siyu Ding and Junyuan Shang and Shuohuan Wang and Yu Sun and Hao Tian and Hua Wu and Haifeng Wang},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2012.15688}
}
```

*Memory is Attention through Time* - Alex Graves
