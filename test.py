import torch
from block_recurrent_transformer_pytorch import BlockRecurrentTransformerEncoder

model = BlockRecurrentTransformerEncoder(
    num_tokens = 20000,             # vocab size
    dim = 768,                      # model dimensions
    intermediate_size = 768,        # Intermediate dim
    depth = 6,                      # depth
    dim_head = 64,                  # attention head dimensions
    heads = 8,                      # number of attention heads
    max_seq_len = 1024,             # the total receptive field of the transformer, in the paper this was 2 * block size
    block_width = 512,              # block size - total receptive field is max_seq_len, 2 * block size in paper. the block furthest forwards becomes the new cached xl memories, which is a block size of 1 (please open an issue if i am wrong)
    xl_memories_layers = (5, 6),    # which layers to use xl memories. very old deepmind papers have shown you only need the last penultimate layers to have cached key values to see majority of benefit
    num_state_vectors = 512,        # number of state vectors, i believe this was a single block size in the paper, but can be any amount
    recurrent_layers = (4,),        # where to place the recurrent layer(s) for states with fixed simple gating
    enhanced_recurrence = True,     # enhanced recurrence from ernie-doc paper, i have seen it to work well on my local machine
    use_flash_attn = True           # use flash attention, if on pytorch 2.0
)

seq = torch.randint(0, 2000, (1, 1024))

out, mems1, states1 = model(seq)
out, mems2, states2 = model(seq, xl_memories = mems1, states = states1)
out, mems3, states3 = model(seq, xl_memories = mems2, states = states2)

print(out.size())

print(model.state_dict().keys())