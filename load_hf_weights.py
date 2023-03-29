import re
from collections import UserDict
from typing import Callable, Dict, List, Union
from torch import nn
from torch import Tensor
from block_recurrent_transformer_pytorch import BlockRecurrentTransformerModel, BlockRecurrentTransformerConfig
from transformers import AutoModel, AutoTokenizer, PretrainedConfig

class ConversionMap(UserDict):
    def __init__(self, map):
        self.string_map = {}
        self.regex_map = {}
        for key, val in map.items():
            if isinstance(key, str):
                self.string_map[key] = val
            elif isinstance(key, re.Pattern):
                self.regex_map[key] = val
            else:
                raise TypeError(f"Encountered invalid key type ({type(key)}) in conversion map key {key}")
        
    def __getitem__(self, key):
        try:
            value = self.string_map[key]
            return value
        except KeyError:
            for pattern, value in self.regex_map.items():
                if (match := re.match(pattern, key)) is not None:
                    formatted_value = value.format(**match.groupdict())
                    return formatted_value
        raise KeyError(f"Key {key} not in ConversionMap")

def print_sizes(model):
    for name, tensor in model.state_dict().items():
        print(name, tensor.size())

def count_params(model):
    n_params = 0
    for param in model.parameters():
        n_params += param.numel()
    return n_params

def convert_config(src_config: PretrainedConfig, translation_map: Dict[str, str], **kwargs) -> BlockRecurrentTransformerConfig:
    """_summary_

    Args:
        translation_map (Dict[str, str]): _description_

    Returns:
        MemorizingTransformerConfig: _description_
    """
    src_config = src_config.to_dict()
    trgt_config = {}

    for src_key, val in src_config.items():
        if (trgt_key := translation_map.get(src_key)) is not None:
            trgt_config[trgt_key] = val
    
    trgt_config |= kwargs
    trgt_config = BlockRecurrentTransformerConfig(**trgt_config)
    return trgt_config


def convert_weights(
        src_model: nn.Module,
        conversion_map: Dict[Union[str, re.Pattern], str],
        additional_funcs: List[Callable] = None
        ) -> Dict[str, Tensor]:
    
    src_state_dict = src_model.state_dict()
    trgt_state_dict= {}
    conversion_map = ConversionMap(conversion_map)
    
    counter = 0
    for src_name, tensor in src_state_dict.items():
        try:
            trgt_name = conversion_map[src_name]
            trgt_state_dict[trgt_name] = tensor
            counter += 1
        except KeyError:
            continue
    
    if additional_funcs is not None:
        for func in additional_funcs:
            trgt_state_dict = func(src_state_dict, trgt_state_dict, counter)
    
    print(f"Converted {counter} entries. (N. Entries of SRCModel = {len(src_state_dict)})")
    return trgt_state_dict


def load_state_dict_merciful(model, state_dict):
    compatible_states = {}
    incompatible_names = []
    for name, tensor in state_dict.items():
        if (model_tensor := model.state_dict().get(name, None)) is not None:
            if tensor.size() == model_tensor.size():
                compatible_states[name] = tensor
            else:
                incompatible_names.append(name)
    model.load_state_dict(compatible_states, strict=False)
    print(f"Could not load the following parameters due to incompatible sizes: {incompatible_names}")
    return model


# HParams of Bert that can be directly translated to MemeTRF-HParams.
BERT_CONFIG_TRANSLATE_MAP = {
    "hidden_size": "dim",
    "vocab_size": "num_tokens",
    "num_hidden_layers": "depth",
    "num_attention_heads": "heads",
    "hidden_dropout_prob": "ff_dropout",
    "intermediate_size": "intermediate_dim"
}
# Rules to convert Bert weights to corresponding memory model weights.
# MemoryTransformers use one-headed keys and values e.g. just use the same query and value vectors for each head of any given layer
# https://github.com/lucidrains/memorizing-transformers-pytorch/issues/5#issuecomment-1163187918
# This means we can't copy query and values vectors from the source model...
BERT_WEIGHT_CONVERSION_MAP = {
    # Token embeddings
    "embeddings.word_embeddings.weight": "encoder.token_emb.weight",
    # AttnLayer query projection
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).attention.self.query.weight"): "encoder.layers.{layer_idx}.0.to_q.weight",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).attention.self.query.bias"): "encoder.layers.{layer_idx}.0.to_q.bias",
    # AttnLayer attention output projection + layer norm
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).attention.output.dense.weight"): "encoder.layers.{layer_idx}.0.to_out.weight",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).attention.output.dense.bias"): "encoder.layers.{layer_idx}.0.to_out.bias",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).attention.output.LayerNorm.weight"): "encoder.layers.{layer_idx}.1.norm.weight",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).attention.output.LayerNorm.bias"): "encoder.layers.{layer_idx}.1.norm.bias",
    # AttnLayer intermediate projections
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).intermediate.dense.weight"): "encoder.layers.{layer_idx}.1.fn.net.0.weight",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).intermediate.dense.bias"): "encoder.layers.{layer_idx}.1.fn.net.0.bias",
    # AttnLayer global output projections + layer norm
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).output.dense.weight"): "encoder.layers.{layer_idx}.1.fn.net.3.weight",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).output.dense.bias"): "encoder.layers.{layer_idx}.1.fn.net.3.bias",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).output.LayerNorm.weight"): "encoder.layers.{layer_idx}.1.norm.weight",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).output.LayerNorm.bias"): "encoder.layers.{layer_idx}.1.norm.bias"
}

if __name__ == "__main__":
   
   # Bert-large
   bert_model = AutoModel.from_pretrained("deepset/gbert-large") 
   bert_tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large")
   print("Bert parameters sizes:")
   print_sizes(bert_model)

   config = convert_config(
       bert_model.config,
       BERT_CONFIG_TRANSLATE_MAP,
       recurrent_layers=(12, 18, 23),
       xl_memory_layers=(6, 14, 20),
       max_seq_len=1024,
       block_width=512,
       enhanced_recurrence=True,
       use_flash_attn=True,
       num_state_vectors=512,
    )
   model = BlockRecurrentTransformerModel(config)
   print("MemoryTransformer parameters sizes:")
   print_sizes(model)

   print("Number of params in source model:", count_params(bert_model))
   print("Number of params in target model:", count_params(model))

   print("Converting weights from Bert to MemoryTransformer")
   converted_state_dict = convert_weights(bert_model, BERT_WEIGHT_CONVERSION_MAP)
   model = load_state_dict_merciful(model, converted_state_dict)

   print("Saving MemoryTransformer")
   model.save_pretrained("_test/recurrent-gbert-large")
   bert_tokenizer.model_max_length = 1024
   bert_tokenizer.init_kwargs["model_max_length"] = 1024
   bert_tokenizer.save_pretrained("_test/recurrent-gbert-large")

   print("Saving model with random initialized weights")
   random_model = BlockRecurrentTransformerModel(config)
   random_model.save_pretrained("_test/rand-recurrent-gbert-large")
   bert_tokenizer.save_pretrained("_test/rand-recurrent-gbert-large")

   #############################################################
   
   # Bert-base
   bert_model = AutoModel.from_pretrained("bert-base-german-cased") 
   bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
   print("Bert parameters sizes:")
   print_sizes(bert_model)
   config = convert_config(
       bert_model.config,
       BERT_CONFIG_TRANSLATE_MAP,
       recurrent_layers=(4,),
       xl_memories_layers=(5, 6),
       max_seq_len=1024,
       block_width=512,
       enhanced_recurrence=True,
       use_flash_attn=True,
       num_state_vectors=512,
    )
   model = BlockRecurrentTransformerModel(config)
   print("MemoryTransformer parameters sizes:")
   print_sizes(model)
   
   print("Number of params in source model:", count_params(bert_model))
   print("Number of params in target model:", count_params(model))
   
   print("Converting weights from Bert to MemoryTransformer")
   converted_state_dict = convert_weights(bert_model, BERT_WEIGHT_CONVERSION_MAP)
   model = load_state_dict_merciful(model, converted_state_dict)
   
   print("Saving MemoryTransformer")
   model.save_pretrained("_test/recurrent-bert-base-german-cased")
   bert_tokenizer.model_max_length = 1024
   bert_tokenizer.init_kwargs["model_max_length"] = 1024
   bert_tokenizer.save_pretrained("_test/recurrent-bert-base-german-cased")
   
   print("Saving model with random initialized weights")
   random_model = BlockRecurrentTransformerModel(config)
   random_model.save_pretrained("_test/rand-recurrent-bert-base-german-cased")
   bert_tokenizer.save_pretrained("_test/rand-recurrent-bert-base-german-cased")



