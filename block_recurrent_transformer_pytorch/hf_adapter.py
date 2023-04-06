from collections import defaultdict
import logging
from inspect import signature
from math import ceil
from typing import Dict, List, Tuple, Union, Optional
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, BatchEncoding
from transformers.data import DataCollatorWithPadding
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput, ModelOutput
from block_recurrent_transformer_pytorch.block_recurrent_transformer_encoder_pytorch import BlockRecurrentTransformerEncoder, LayerNorm

logger = logging.getLogger(__name__)

class BlockRecurrentTransformerConfig(PretrainedConfig):
    def __init__(
            self,
            max_train_segments = 5,
            max_eval_segments = None,
            num_tokens = 20_000,
            dim = 768,
            depth = 12,
            dim_head = 64,
            intermediate_dim = 4096,
            heads = 12,
            all_layers_qk_rmsnorm = False,
            max_seq_len = 1024,
            block_width = 512,
            xl_memories_layers: Optional[Tuple[int, ...]] = None,
            recurrent_layers: Optional[Tuple[int, ...]] = None,
            num_state_vectors = None,
            enhanced_recurrence = False,
            ignore_index = -100,
            use_flash_attn = False,
            pad_segments = False,
            gate_type = "fixed",
            position_encoding_type = "rel_bias",
            ff_dropout = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.max_train_segments = max_train_segments
        self.max_eval_segments = max_eval_segments
        self.num_tokens = num_tokens
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.intermediate_dim = intermediate_dim
        self.heads = heads
        self.all_layers_qk_rmsnorm = all_layers_qk_rmsnorm
        self.block_width = block_width
        self.xl_memories_layers = xl_memories_layers
        self.recurrent_layers = recurrent_layers
        self.num_state_vectors = num_state_vectors
        self.enhanced_recurrence = enhanced_recurrence
        self.ignore_index = ignore_index
        self.use_flash_attn = use_flash_attn
        self.pad_segments = pad_segments # torch.compile requires fixed size batches...
        self.gate_type = gate_type
        self.position_encoding_type = position_encoding_type
        self.ff_dropout = ff_dropout


class BlockRecurrentTransformerModel(PreTrainedModel):
    
    MODEL_OUTPUT = BaseModelOutput
    
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # If the config is read from disk, tuples become lists and are not hashable anymore...
        config = {arg: val if not isinstance(val, list) else tuple(val) for arg, val in config.to_dict().items()}
        
        self.encoder = BlockRecurrentTransformerEncoder(**{
            arg: val
            for arg, val in config.items()
            if arg in signature(BlockRecurrentTransformerEncoder.__init__).parameters
        })        
            

    def forward(self, *args, **kwargs) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        outputs = self._forward(*args, **kwargs)
        if isinstance(outputs, dict):
            return self.MODEL_OUTPUT(**outputs)
        else:
            return outputs
    
    def forward_segment(
            self,
            input_ids: torch.Tensor,
            states = None,
            xl_memories = None,
            return_dict: bool = True,
            *args,
            **kwargs
        ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        if states is None: states = []
        if xl_memories is None: xl_memories = []
        embeddings, last_xl_memories, last_states = self.encoder(
            x=input_ids,
            states=states,
            xl_memories=xl_memories,
            return_memories_and_states=True
        )

        if return_dict:
            outputs = dict(last_hidden_state=embeddings)
        else:
            outputs = (embeddings,)
        return outputs, last_xl_memories, last_states
    
    def _forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        
        if labels is not None:
            inputs = BatchEncoding(dict(input_ids=input_ids, labels=labels, **kwargs))
        else:
            inputs = BatchEncoding(dict(input_ids=input_ids, **kwargs))
        
        segments = self._split_batch_into_segments(inputs)
        
            
        if self.training and self.config.max_train_segments is not None:
            segments = segments[:self.config.max_train_segments]
        elif not self.training and self.config.max_eval_segments is not None:
            segments = segments[:self.config.max_eval_segments]
        
        last_states = None
        last_xl_memories = None
        segment_outputs = []
        for segment in segments:
            segment["states"] = last_states
            segment["xl_memories"] = last_xl_memories
            segment["return_dict"] = return_dict
            segment_output, last_xl_memories, last_states = self.forward_segment(**segment)
            segment_outputs.append(segment_output)

        if return_dict:
            outputs = self._gather_results_from_dicts(segment_outputs)
        else:
            outputs = self._gather_results_from_tuples(segment_outputs)

        return outputs
        
    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.token_emb
    
    # def _split_batch_into_segments(self, inputs: BatchEncoding) -> List[BatchEncoding]:
    #     """
    #     Training a block recurrent transformer requires a special batching of texts,
    #     because it is crucial to process longer texts segment-wise 
    #     in order to train or - during inference - leverage the memory.

    #     This function expects batches to be padded to max_length.

    #     It takes a batch of texts in (padded) original length and 
    #     splits them into round(batch_text_length // max_seq_length) segments.
    #     """

    #     text_length = inputs["input_ids"].size(1)
    #     # Round UP to next common denominator, to avoid overflowing segments...
    #     text_length_rounded = self.config.max_seq_len * ceil(text_length / self.config.max_seq_len)
    #     n_segments = max(round(text_length_rounded // self.config.max_seq_len), 1)
        
    #     segments = [{} for _ in range(n_segments)]
    #     for key, tensor in inputs.items():
    #         segmented_tensors = tensor.chunk(n_segments, dim=-1) 
    #         for idx, (segment, segmented_tensor) in enumerate(zip(segments, segmented_tensors)):
    #             segmented_tensor = segmented_tensor.contiguous()
    #             if idx == 0 and (seg_len := segmented_tensor.size(-1)) > self.config.max_seq_len:
    #                 logger.warning(f"Encountered segment with invalid length ({seg_len})")
    #             segment[key] = segmented_tensor
    #     segments = [BatchEncoding(segment) for segment in segments]
    #     if self.config.pad_segments:
    #         segments = self.collator(segments)
    #     return segments
     
    def _split_batch_into_segments(self, inputs: BatchEncoding) -> List[BatchEncoding]:
        max_seq_len = self.config.max_seq_len
        chunked_data = defaultdict(list)
        for key, tensor in inputs.items():
            splitted_tensors = tensor.split(max_seq_len, dim=1)
            chunked_data[key].extend(splitted_tensors)
        segments = [
            BatchEncoding({key: chunks[i] for key, chunks in chunked_data.items()})
            for i in range(len(chunked_data["input_ids"]))
        ]
        return segments
        



    @staticmethod
    def _gather_results_from_dicts(segment_outputs: List[ModelOutput]) -> Dict[str, torch.Tensor]:
        leader_output = segment_outputs[0]        
        all_outputs = defaultdict(list)
        for key in leader_output.keys():
            for segment_output in segment_outputs:
                all_outputs[key].append(segment_output[key])
        
        concatenated_outputs = dict(
            (key, torch.cat(tensors, dim=1)) if key != "loss" else (key, sum(tensors) / len(tensors))
            # This version only keeps the last loss to pass it to trainer because the others were already backpropagated.
            # (key, torch.cat(tensors, dim=1)) if key != "loss" else (key, tensors[-1] / len(all_outputs))
            for key, tensors in all_outputs.items()
        )
        return concatenated_outputs
    
    @staticmethod
    def _gather_results_from_tuples(segment_outputs: List[Tuple[torch.Tensor]], labels: torch.Tensor = None) -> Tuple[torch.Tensor]:
        # Tuple order ([loss], <logits/embs>, <rest>)
        leader_output = segment_outputs[0]
        gathered_outputs = [[] for _ in range(len(leader_output))]
        for segment_output in segment_outputs:
            for idx, entry in enumerate(segment_output):
                gathered_outputs[idx].append(entry)
        if labels is not None:
            all_losses = gathered_outputs.pop(0)
            # This version only keeps the last loss to pass it to trainer because the others were already backpropated.
            # loss = all_losses[-1] / len(gathered_outputs)
            loss = sum(all_losses) / len(all_losses)
        outputs = tuple(torch.cat(entries, dim=1) for entries in gathered_outputs)
        if labels is not None:
            outputs = (loss,) + outputs
        return outputs
    
    def init_segment_collator(self, tokenizer):
        self.collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="max_length",
            max_length=self.config.max_seq_len
        )
    
class BlockRecurrentTransformerForMaskedLM(BlockRecurrentTransformerModel):
    
    MODEL_OUTPUT = MaskedLMOutput

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.to_classify = nn.Sequential(
            LayerNorm(self.config.dim),
            nn.Linear(self.config.dim, self.config.dim),
            nn.GELU(),
            LayerNorm(self.config.dim),
            nn.Dropout(self.config.ff_dropout)
        )

        self.classifier = nn.Linear(self.config.dim, self.config.num_tokens, bias=False)

    def forward_segment(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor = None,
            states = None,
            xl_memories = None,
            return_dict: bool = True,
            *args,
            **kwargs
        ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        outputs, last_xl_memories, last_states = super().forward_segment(
            input_ids=input_ids,
            states=states,
            xl_memories=xl_memories,
            return_dict=return_dict,
            *args, **kwargs
        )
        
        if isinstance(outputs, dict):
            embeddings = outputs["last_hidden_state"]
        else:
            embeddings = outputs[0]
        embeddings = self.to_classify(embeddings)
        logits = self.classifier(embeddings)
        
        if labels is not None:
            *_, num_tokens = logits.size()
            loss = F.cross_entropy(logits.reshape(-1, num_tokens), labels.reshape(-1))
            # Due to chunking, we sometimes encounter segments without any masked-out tokens.
            # In these cases the loss is NaN, and we replace it with a artificial zero loss
            if torch.isnan(loss):
                warnings.warn("Encountered NaN loss in LMHead.")
                loss = torch.tensor(
                    0.0,
                    requires_grad=True,
                    dtype=logits.dtype,
                    device=loss.device
                )
        
        if return_dict:
            outputs = MaskedLMOutput(logits=logits)
            if labels is not None:
                outputs["loss"] = loss
        else:
            outputs = (logits,)
            if labels is not None:
                outputs = (loss,) + outputs
        
        return outputs, last_xl_memories, last_states
        
