#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2023 Yanwei Li
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertLMHeadModel as BertLMHeadModelRaw

from .qformer import BertConfig
from .qformer import BertLMHeadModel as BertLMHeadModelQF

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llamavid.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LLaMAVIDMetaModel:

    def __init__(self, config):
        super(LLaMAVIDMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None, max_token=2048):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.image_processor = getattr(model_args, 'image_processor', None)

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.max_token = max_token
        
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_attention_modules(self, model_args, for_eval=False):  
        pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        pretrain_qformer = getattr(model_args, "pretrain_qformer", None)
        self.config.bert_type = getattr(model_args, "bert_type", "qformer")
        self.config.num_query = getattr(model_args, "num_query", 32)
        self.config.compress_type = getattr(model_args, "compress_type", None)

        if 'pretrain' in self.config.bert_type:
            # for qformer that use evaclip for prtrain
            att_feat_size = 1408
        else:
            att_feat_size = self.config.mm_hidden_size
        self.vlm_att_tokenlizer, self.vlm_att_encoder, self.vlm_att_query = self.init_bert(att_feat_size, truncation_side="left")
        self.vlm_att_projector = torch.nn.Linear(self.vlm_att_encoder.config.hidden_size, self.config.mm_hidden_size)
        self.vlm_att_key_projector  = torch.nn.Linear(self.config.mm_hidden_size, self.config.mm_hidden_size)
        self.vlm_att_val_projector  = torch.nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)

        if "raw" in self.config.bert_type:
            self.vlm_att_bert_proj  = torch.nn.Linear(att_feat_size, self.vlm_att_encoder.config.hidden_size)
        elif "pretrain" in self.config.bert_type and self.config.mm_hidden_size!=att_feat_size:
            self.vlm_att_bert_proj = torch.nn.Linear(self.config.mm_hidden_size, att_feat_size)
        else:
            self.vlm_att_bert_proj = None
        
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        
        if 'qformer_pretrain' in self.config.bert_type:
            self.vlm_att_ln = torch.nn.LayerNorm(att_feat_size)
        
        if pretrain_qformer is not None:
            print("Loading pretrained qformer weights...")
            qformer_weight = torch.load(pretrain_qformer, map_location='cpu')['model']
            bert_weight = {_key: qformer_weight[_key] for _key in qformer_weight if 'bert' in _key}
            self.vlm_att_encoder.load_state_dict(get_w(bert_weight, 'Qformer'))
            self.vlm_att_ln.load_state_dict(get_w(qformer_weight, 'ln_vision'))
            self.vlm_att_query.data = qformer_weight['query_tokens']
        
        if 'freeze_all' in self.config.bert_type:
            print("Freezing all qformer weights...")
            self.vlm_att_encoder.requires_grad_(False)
            self.vlm_att_ln.requires_grad_(False)
            self.vlm_att_query.requires_grad_(False)
            self.vlm_att_projector.requires_grad_(False)
            self.vlm_att_key_projector.requires_grad_(False)
            self.vlm_att_val_projector.requires_grad_(False)
        elif 'freeze' in self.config.bert_type:
            print("Freezing pretrained qformer weights...")
            self.vlm_att_encoder.requires_grad_(False)
            self.vlm_att_ln.requires_grad_(False)
            self.vlm_att_query.requires_grad_(False)
        

        if pretrain_mm_mlp_adapter is not None:
            att_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        else:
            trainable_module = ['vlm_att_encoder', 'vlm_att_projector', 'vlm_att_key_projector', 
                                'vlm_att_val_projector', 'vlm_att_query', 'vlm_att_visual_proj',
                                'vlm_att_ln']
            if hasattr(model_args, 'model_name_or_path'):
                model_save_path = model_args.model_name_or_path
            else:
                model_save_path = model_args.model_path
            model_idx_path = getattr(model_args, 'model_path', model_save_path)
            weight_file = json.load(open(os.path.join(model_idx_path, 'pytorch_model.bin.index.json'), 'r'))['weight_map']
            model_path = set([weight_file[_key] for _key in weight_file if any([_module in _key for _module in trainable_module])])
            att_projector_weights = {}
            for _model in model_path:
                att_projector_weights.update(torch.load(os.path.join(model_idx_path, _model), map_location='cpu'))
            if len(att_projector_weights) == 0:
                return
        
        bert_dict = get_w(att_projector_weights, 'vlm_att_encoder')
        if "bert.embeddings.position_ids" not in bert_dict and "raw_bert" not in self.config.bert_type:
            bert_dict["bert.embeddings.position_ids"] = self.vlm_att_encoder.bert.embeddings.position_ids
        print('Loading pretrained weights...')
        self.vlm_att_encoder.load_state_dict(bert_dict)
        self.vlm_att_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_projector'))
        self.vlm_att_key_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_key_projector'))
        self.vlm_att_val_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_val_projector'))

        if "qformer" in self.config.bert_type:
            print('Loading vlm_att_query weights...')
            self.vlm_att_query.data = att_projector_weights['model.vlm_att_query']
            if "pretrain" in self.config.bert_type:
                print('Loading vlm_att_ln weights...')
                self.vlm_att_ln.load_state_dict(get_w(att_projector_weights, 'vlm_att_ln'))

        if self.vlm_att_bert_proj is not None:
            print('Loading vlm_att_bert_proj weights...')
            self.vlm_att_bert_proj.load_state_dict(get_w(att_projector_weights, 'vlm_att_bert_proj'))
        
        if for_eval:
            weight_type = torch.float16
            device_type = self.mm_projector[0].weight.device
            self.vlm_att_encoder = self.vlm_att_encoder.to(device=device_type, dtype=weight_type)
            self.vlm_att_projector = self.vlm_att_projector.to(device=device_type, dtype=weight_type)
            self.vlm_att_key_projector = self.vlm_att_key_projector.to(device=device_type, dtype=weight_type)
            self.vlm_att_val_projector = self.vlm_att_val_projector.to(device=device_type, dtype=weight_type)

            if "qformer" in self.config.bert_type:
                self.vlm_att_query.data = self.vlm_att_query.data.to(device=device_type, dtype=weight_type)
                if "pretrain" in self.config.bert_type:
                    self.vlm_att_ln = self.vlm_att_ln.to(device=device_type, dtype=weight_type)
            
            if self.vlm_att_bert_proj is not None:
                self.vlm_att_bert_proj = self.vlm_att_bert_proj.to(device=device_type, dtype=weight_type)
            

    def init_bert(self, vision_width, cross_attention_freq=2, truncation_side="right"):
        # initialize BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # initialize BERT
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        query_tokens = None
        
        if "qformer" in self.config.bert_type:
            mm_model = BertLMHeadModelQF.from_pretrained(
                "bert-base-uncased", config=encoder_config
            )
            query_tokens = nn.Parameter(
                torch.zeros(1, self.config.num_query, encoder_config.hidden_size)
            )
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        elif "raw" in self.config.bert_type:
            encoder_config.is_decoder = True
            mm_model = BertLMHeadModelRaw.from_pretrained(
                "bert-base-uncased", config=encoder_config
            )
        else:
            raise NotImplementedError("BERT type not implemented...")
        
        mm_model.resize_token_embeddings(len(tokenizer))
        mm_model.cls = None
        
        if "layer" in self.config.bert_type:
            layer_num = int(self.config.bert_type.split(':')[-1])
            mm_model.bert.encoder.layer = mm_model.bert.encoder.layer[:layer_num]
            print(f"Only use {layer_num} layers in BERT...")
        
        return tokenizer, mm_model, query_tokens


class LLaMAVIDMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, prompts=None, image_counts=None, long_video=False):        
        if long_video:
            # use pre-computed features
            image_features = images
        else:
            image_features = self.get_model().get_vision_tower()(images)

        image_features = self.vlm_attention(image_features, 
                                            prompts=prompts, 
                                            image_counts=image_counts,
                                            long_video=long_video)
        return image_features

    def vlm_attention(self, image_features, prompts=None, image_counts=None, long_video=False):        
        img_feat_lst = []
        if image_counts is None:
            assert len(image_features) == len(prompts), f"Size mismatch! image_features: {len(image_features)}, prompts: {len(prompts)}"
        else:
            assert len(prompts) == len(image_counts), f"Size mismatch! prompts: {len(prompts)}, image_counts: {len(image_counts)}"
        image_atts = torch.ones(image_features.size()[:-1], dtype=torch.long).to(image_features.device)    

        total_count = 0
        # calculate each image feat according to the prompt
        for _idx in range(len(prompts)):
            assert isinstance(prompts[_idx], list), f"Prompt should be a list, but got {type(prompts[_idx])}"
            input_token = self.get_model().vlm_att_tokenlizer(
                prompts[_idx], 
                padding='longest', 
                truncation=True,
                max_length=256,
                return_tensors="pt"
                ).to(image_features.device)

            input_ids = input_token.input_ids
            attention_masks = input_token.attention_mask
            
            if image_counts is None:
                img_feat_prompt = image_features[_idx, None].expand(len(prompts[_idx]), -1, -1)
                img_att_prompt = image_atts[_idx, None].expand(len(prompts[_idx]), -1)
            else:
                # shape: [prompt_num*frame_num, image_shape, feat_dim]
                img_feat_prompt = image_features[total_count:total_count+image_counts[_idx]]
                img_feat_prompt = img_feat_prompt[None].expand(len(prompts[_idx]), -1, -1, -1).flatten(0,1)
                img_att_prompt = image_atts[total_count:total_count+image_counts[_idx]]
                img_att_prompt = img_att_prompt[None].expand(len(prompts[_idx]), -1, -1).flatten(0,1)
                input_ids = input_ids[:,None].expand(-1, image_counts[_idx], -1).flatten(0,1)
                attention_masks = attention_masks[:,None].expand(-1, image_counts[_idx], -1).flatten(0,1)
                total_count += image_counts[_idx]
            
            if "pretrain" in self.config.bert_type and self.get_model().vlm_att_bert_proj is not None:
                bert_feat = self.get_model().vlm_att_bert_proj(img_feat_prompt)
            else:
                bert_feat = img_feat_prompt.clone()

            # remove cls embedding
            if self.config.mm_vision_select_feature == 'patch':
                if img_feat_prompt.shape[1]%2 == 1:
                    img_feat_prompt = img_feat_prompt[:, 1:]

            if "qformer" in self.config.bert_type:
                query_tokens = self.get_model().vlm_att_query.expand(bert_feat.shape[0], -1, -1)
                query_atts = torch.cat([torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(bert_feat.device), 
                                        attention_masks],dim=1)
                
                if 'pretrain' in self.config.bert_type:
                    mm_img_in = self.get_model().vlm_att_ln(bert_feat)
                else:
                    mm_img_in = bert_feat
                
                if long_video:
                    outputs = []
                    block_size = 64
                    for L in range(0, len(input_ids), block_size):
                        R = L + block_size
                        mm_output = self.get_model().vlm_att_encoder.bert(
                            input_ids[L:R],
                            query_embeds=query_tokens[L:R],
                            attention_mask=query_atts[L:R],
                            encoder_hidden_states=mm_img_in[L:R],
                            encoder_attention_mask=img_att_prompt[L:R],
                            return_dict=True,
                        )
                        mm_output = mm_output.last_hidden_state[:,:query_tokens.shape[1]]
                        outputs.append(mm_output)
                    mm_output = torch.cat(outputs)
                    torch.cuda.empty_cache()
                else:
                    mm_output = self.get_model().vlm_att_encoder.bert(
                        input_ids,
                        query_embeds=query_tokens,
                        attention_mask=query_atts,
                        encoder_hidden_states=mm_img_in,
                        encoder_attention_mask=img_att_prompt,
                        return_dict=True,
                    )
                    mm_output = mm_output.last_hidden_state[:,:query_tokens.shape[1]]
                
            elif "raw" in self.config.bert_type:
                if self.config.mm_vision_select_feature == 'patch' and bert_feat.shape[1]%2 == 1:
                    bert_feat = bert_feat[:, 1:]
                    img_att_prompt = img_att_prompt[:, 1:]
                
                mm_output = self.get_model().vlm_att_encoder.bert(
                    input_ids,
                    attention_mask=attention_masks,
                    encoder_hidden_states=self.get_model().vlm_att_bert_proj(bert_feat),
                    encoder_attention_mask=img_att_prompt,
                    return_dict=True,
                )
                mm_output = mm_output.last_hidden_state
            else:
                raise ValueError(f'Unexpected bert type: {self.config.bert_type}')
            
            text_q = self.get_model().vlm_att_projector(mm_output)
            final_token = self.token_generation(text_q, img_feat_prompt, long_video=long_video)

            if image_counts is not None:
                # shape: [prompt_num, frame_num*image_shape, feat_dim]
                final_token = final_token.reshape(len(prompts[_idx]), image_counts[_idx], *final_token.shape[-2:])
                final_token = final_token.flatten(1,2)
            img_feat_lst.append(final_token)

        return img_feat_lst

    def token_generation(self, text_q, vis_embed, long_video=False):
        ctx_embed = self.get_model().vlm_att_key_projector(vis_embed)
        # Key part 1: calculate context-related embedding
        ctx_embed = text_q @ ctx_embed.transpose(-1,-2) 
        ctx_embed = ctx_embed / (vis_embed.shape[-1] ** 0.5)
        if not long_video:
            ctx_embed = (ctx_embed.softmax(-1) @ vis_embed).mean(1)
        else:
            block_size = 64
            outputs = []
            ctx_score = ctx_embed.softmax(-1)    
            for L in range(0, len(ctx_score), block_size):
                R = L + block_size
                sub_embed = (ctx_score[L:R] @ vis_embed[L:R]).mean(1)
                outputs.append(sub_embed)
            ctx_embed = torch.cat(outputs)
            torch.cuda.empty_cache()
        ctx_embed = self.get_model().vlm_att_val_projector(ctx_embed[:,None])

        # Key part 2: calculate visual embedding
        if self.config.compress_type is not None:
            if 'grid' in self.config.compress_type:
                grid_size = int(self.config.compress_type.split('grid:')[-1])
                cur_shape = int(vis_embed.shape[1]**0.5)
                assert grid_size > 1, f'Grid size should be larger than 1, but got {grid_size}'
                vis_embed = vis_embed.reshape(vis_embed.shape[0], cur_shape, cur_shape, -1)
                grid_stride = cur_shape // grid_size
                vis_embed = F.avg_pool2d(vis_embed.permute(0, 3, 1, 2), 
                                         padding=0,
                                         kernel_size=grid_stride, 
                                         stride=grid_stride)
                
                vis_embed = vis_embed.permute(0, 2, 3, 1).flatten(1,2)
            elif 'mean' in self.config.compress_type:
                vis_embed = vis_embed.mean(dim=1, keepdim=True)
        
        # concat token in shape (B, n+1, C)
        vis_embed = self.get_model().mm_projector(vis_embed)                
        final_token = torch.cat([ctx_embed, vis_embed], dim=1)
        return final_token

    def update_prompt(self, prompts=None):
        self.prompts = prompts

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, prompts=None
    ): 
        if prompts is None and hasattr(self, 'prompts'):
            prompts = self.prompts
        
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels
        
        # pre-process images for long video
        if images[0].shape[-1] > 1000:
            long_video = True
        else:
            long_video = False

        if type(images) is list or images.ndim == 5:
            # not reseshape for long video
            if not long_video:
                images = [image if len(image.shape) == 4 else image.unsqueeze(0) for image in images]
            image_counts = [image.shape[0] for image in images]
            concat_images = torch.cat(images, dim=0)
            image_features = self.encode_images(concat_images, prompts, image_counts, long_video=long_video)
        else:
            image_features = self.encode_images(images, prompts, long_video=long_video)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                if isinstance(image_features, list):
                    cur_image_features = image_features[cur_image_idx][0]
                else:
                    cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            
            if not long_video:
                token_idx = 0
                while image_token_indices.numel() > 0:
                    if isinstance(image_features, list):
                        cur_image_features = image_features[cur_image_idx][token_idx]
                    else:
                        cur_image_features = image_features[cur_image_idx]
                    image_token_start = image_token_indices[0]
                    
                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                            cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                            cur_labels = cur_labels[image_token_start+2:]
                    else:
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                            cur_labels = cur_labels[image_token_start+1:]
                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                        cur_input_ids = cur_input_ids[image_token_start+2:]
                    else:
                        cur_input_ids = cur_input_ids[image_token_start+1:]
                    image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                    token_idx += 1
                
                # changle image idx after processing one sample
                cur_image_idx += 1
                if cur_input_ids.numel() > 0:
                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                    else:
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                    if labels is not None:
                        cur_new_labels.append(cur_labels)
                cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
                cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    cur_new_labels = torch.cat(cur_new_labels, dim=0)
                    new_labels.append(cur_new_labels)
            else:
                cur_new_input_embeds = torch.Tensor(len(cur_input_ids), self.config.hidden_size).to(dtype=self.dtype, device=self.device)
                text_token_indices = torch.where(cur_input_ids != IMAGE_TOKEN_INDEX)[0]
                if not self.training and self.get_model().embed_tokens.weight.device != cur_input_ids.device:
                    model_device = self.get_model().embed_tokens.weight.device
                    data_device = cur_input_ids.device
                    cur_input_ids_text = cur_input_ids[text_token_indices].to(device=model_device)
                    cur_new_input_embeds[text_token_indices] = self.get_model().embed_tokens(cur_input_ids_text).to(device=data_device)
                else:
                    cur_new_input_embeds[text_token_indices] = self.get_model().embed_tokens(cur_input_ids[text_token_indices])
                cur_image_features = image_features[cur_image_idx]
                cur_new_input_embeds[image_token_indices] = cur_image_features
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    new_labels.append(cur_labels)
                cur_image_idx += 1

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
