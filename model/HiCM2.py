"""
HiCM^2 backbone with PMD modifications.

Modifications relative to the official HiCM^2 implementation:
  - VMA (Visual Modality Attenuation): training-time visual feature scaling
    by lambda. With --use_vma, lambda is set to 0 throughout Phase 3 training,
    and is implicitly restored to 1 at inference (the scaling block runs only
    under self.training).
  - SaliGT (Phase 2): the existing saliency reweighting hook is preserved.
  - STT (Phase 3): handled at the data side (dvc_dataset.py); no model change.

Note: the variable name `use_vma` reflects the original code lineage; the
paper refers to this mechanism as VMA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from transformers import T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

from .modeling_t5 import T5ForConditionalGeneration
from .vit import VisionTransformer


def _get_tokenizer(tokenizer_path, num_bins=0):
    if 't5' in tokenizer_path:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if num_bins:
            new_tokens = ["<time=" + str(i) + ">" for i in range(num_bins)]
            tokenizer.add_tokens(list(new_tokens))
    else:
        raise NotImplementedError(tokenizer_path)
    return tokenizer


def dist_is_main():
    """Safe check for main process."""
    try:
        import torch.distributed as dist_module
        if dist_module.is_initialized():
            return dist_module.get_rank() == 0
    except Exception:
        pass
    return True


class Vid2Seq(torch.nn.Module):
    """
    HiCM^2 backbone (T5-Base + CLIP ViT-L/14 + hierarchical retrieval).

    PMD additions are kept minimal and gated by args flags:
      - args.use_vma            : VMA, training-time visual scaling (lambda=0)
      - args.use_saliency_reweight : Phase 2 SaliGT (consumes
                                     batch['saliency_weights'] from dataset)
    """

    def __init__(self,
                 t5_path,
                 num_features=100,
                 embed_dim=768,
                 depth=12,
                 heads=12,
                 mlp_dim=2048,
                 vis_drop=0.,
                 tokenizer=None,
                 enc_drop=0.,
                 dec_drop=0.1,
                 use_speech=True,
                 use_video=True,
                 num_bins=100,
                 label_smoothing=0.1,
                 memory_bank=None,
                 args=None):
        super().__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            encoder_dropout=enc_drop,
            decoder_dropout=dec_drop,
            label_smoothing=label_smoothing,
            pretrained_model_name_or_path=t5_path,
            local_files_only=True,
            is_gated_act="v1_1" in t5_path,
            args=args,
        )
        self.t5_model.resize_token_embeddings(len(tokenizer) - num_bins)
        self.t5_model.resize_token_embeddings(len(tokenizer))

        self.visual_encoder = VisionTransformer(
            num_features=num_features,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=heads,
            mlp_dim=mlp_dim,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=vis_drop,
            attn_drop_rate=vis_drop,
            norm_layer=nn.LayerNorm,
        )

        self.t5_tokenizer = tokenizer
        self.use_speech = use_speech
        self.use_video = use_video

        self.proj_v2t = None
        if self.t5_model.model_dim != 768:
            self.proj_v2t = nn.Linear(768, self.t5_model.model_dim)

        # ---------------- HiCM^2 retrieval branch ----------------
        self.memory_bank = memory_bank
        self.args = args

        if self.memory_bank is not None:
            if args.ret2t5_proj == "deep":
                n_input_proj = 2
                txt_dim = 768
                hidden_dim = 768
                input_dropout = 0.5
                self.n_input_proj = n_input_proj
                relu_args = [True] * 3
                relu_args[n_input_proj - 1] = False
                self.ret2t5_proj = nn.Sequential(*[
                    LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
                    LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
                    LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2]),
                ][:n_input_proj])
            elif args.ret2t5_proj == "simple":
                self.ret2t5_proj = nn.Linear(768, self.t5_model.model_dim)

        # ---------------- VMA (training-time visual attenuation) ----------------
        # When --use_vma is set, visual features are scaled by 0 throughout
        # training. At inference, this block is bypassed (self.training is False),
        # so the full visual signal is always used at evaluation time.
        self.use_vma = getattr(args, 'use_vma', False)
        self.vma_lambda = getattr(args, 'vma_lambda', 0.0)
    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(self, video, input_tokenized, output_tokenized, mode='None',
                uns_video=None, **kwargs):
        # Retrieval lookup (HiCM^2 hierarchical memory)
        if self.memory_bank is not None:
            if isinstance(video, dict):
                target_video = video["video"]
            else:
                target_video = video
            ret_texts = self.ret(target_video, self.memory_bank, mode, uns_video=uns_video)

        # Text branch (ASR + appended STT tokens, if any)
        if self.use_speech:
            text = self.t5_model.encoder.embed_tokens(input_tokenized['input_ids'])
            encoded = self.t5_model.encoder(
                attention_mask=input_tokenized['attention_mask'],
                inputs_embeds=text,
            )

        # Visual branch
        if self.use_video:
            if isinstance(video, dict):
                video, atts_vis = video["video"], video["atts_vis"]
            else:
                video = self.visual_encoder(video)
                if self.proj_v2t is not None:
                    video = self.proj_v2t(video)

                # ===== VMA: training-time visual scaling by lambda=0 =====
                # Implementation note: scaling by exactly 0 is intentional and
                # matches the paper's Phase 3 setup. Inference bypasses this
                # block via the self.training check below.
                if self.use_vma and self.training:
                    video = video * self.vma_lambda

                # ===== Phase 2 SaliGT: GT saliency reweighting =====
                if getattr(self.args, 'use_saliency_reweight', False):
                    saliency_weights = kwargs.get('saliency_weights', None)
                    if saliency_weights is not None:
                        video = video * saliency_weights.unsqueeze(-1)

                atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)

            # Concatenate visual + retrieval features (HiCM^2)
            if self.args.ret_option == "hier_concat":
                video = torch.cat([video, ret_texts], dim=1)
                atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)

            video_dict = {"video": video, "atts_vis": atts_vis}
        else:
            video_dict = None

        # Combine video + text for the T5 encoder context
        if self.use_video and self.use_speech:
            encoded.last_hidden_state = torch.cat([video, encoded.last_hidden_state], dim=1)
            encoder_atts = torch.cat([atts_vis, input_tokenized['attention_mask']], dim=1)
        elif self.use_video:
            encoded = BaseModelOutput(last_hidden_state=video)
            encoder_atts = atts_vis
        elif self.use_speech:
            encoder_atts = input_tokenized['attention_mask']

        targets = output_tokenized['input_ids'].masked_fill(
            output_tokenized['input_ids'] == self.t5_tokenizer.pad_token_id, -100
        )

        outputs = self.t5_model(
            encoder_outputs=encoded,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokenized['attention_mask'],
            return_dict=True,
            labels=targets,
        )

        return {"loss": outputs.loss}, video_dict

    # ------------------------------------------------------------------ #
    #  Generate
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        video,
        input_tokenized,
        use_nucleus_sampling=False,
        num_beams=4,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        uns_video=None,
        mode=None,
    ):
        # Retrieval lookup
        if self.memory_bank is not None:
            ret_texts = self.ret(video, self.memory_bank, mode, uns_video=uns_video)

        # Text branch
        if self.use_speech:
            text = self.t5_model.encoder.embed_tokens(input_tokenized['input_ids'])
            encoded = self.t5_model.encoder(
                attention_mask=input_tokenized['attention_mask'],
                inputs_embeds=text,
            )

        # Visual branch (no VMA at inference; no SaliGT at inference)
        if self.use_video:
            if isinstance(video, dict):
                video, atts_vis = video["video"], video["atts_vis"]
            else:
                video = self.visual_encoder(video)
                if self.proj_v2t is not None:
                    video = self.proj_v2t(video)
                atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)

            # NOTE: at generation, the original HiCM^2 puts retrieved tokens
            # before the visual tokens (different order from forward); preserved
            # here to match the upstream behavior.
            if self.args.ret_option == "hier_concat":
                video = torch.cat([ret_texts, video], dim=1)
                atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)

            video_dict = {"video": video, "atts_vis": atts_vis}
        else:
            video_dict = None

        if self.use_video and self.use_speech:
            encoded.last_hidden_state = torch.cat([video, encoded.last_hidden_state], dim=1)
            encoder_atts = torch.cat([atts_vis, input_tokenized['attention_mask']], dim=1)
        elif self.use_video:
            encoded = BaseModelOutput(last_hidden_state=video)
            encoder_atts = atts_vis
        elif self.use_speech:
            encoder_atts = input_tokenized['attention_mask']

        outputs = self.t5_model.generate(
            encoder_outputs=encoded,
            attention_mask=encoder_atts,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text

    # ------------------------------------------------------------------ #
    #  Retrieval (HiCM^2 official)
    # ------------------------------------------------------------------ #
    def softattention_select(self, memory_bank, feature, mode, uns_video=None):
        soft_k = self.args.soft_k
        if self.args.ret_option in ("hier_ret", "hier_concat"):
            if self.args.sim_match == "anchor_cos":
                window_size = self.args.window_size
                frame_length = feature.shape[1]

                topk_window_embeds = []
                total_window_sents = []

                for b in range(feature.shape[0]):
                    batch_topk_window_embeds = []
                    segment_length = frame_length // window_size
                    for i in range(window_size):
                        start = i * segment_length
                        end = start + segment_length
                        target_feature = torch.mean(feature[b, start:end], dim=0)
                        topk_embed = self.hierarchical_memory_search(
                            target_feature, soft_k, memory_bank
                        )
                        batch_topk_window_embeds.append(topk_embed)

                    batch_topk_window_embeds = torch.cat(
                        batch_topk_window_embeds, dim=0
                    ).unsqueeze(0).float()
                    topk_window_embeds.append(batch_topk_window_embeds)
                    total_window_sents.append('no')

                topk_window_embeds = torch.cat(topk_window_embeds, dim=0)
                return topk_window_embeds, total_window_sents

    def get_topk_indices(self, similarity_scores, k):
        if similarity_scores.shape[0] < k:
            return torch.arange(similarity_scores.shape[0])
        return torch.topk(similarity_scores, k, dim=0).indices

    def hierarchical_memory_search(self, target_feature, soft_k, memory_hierarchy):
        k = soft_k
        threshold = 0.7
        selected_levels = self.args.hier_use
        retrieval_type = self.args.hier_ret_num

        combined_vectors = []
        topk_clusters = []

        sorted_levels = sorted(memory_hierarchy.keys(),
                               key=lambda x: int(x.split('_')[1]),
                               reverse=True)

        level_summaries = {level: [] for level in sorted_levels}
        for i, level in enumerate(sorted_levels):
            clusters = memory_hierarchy[level]
            if not topk_clusters:
                topk_clusters = [
                    (cosine_similarity(target_feature.unsqueeze(0), cluster["clip_embedding"]), cluster)
                    for cluster_id, cluster in clusters.items()
                ]
                topk_clusters.sort(key=lambda x: x[0], reverse=True)
                if retrieval_type == "max":
                    topk_clusters = topk_clusters[:1]
                elif retrieval_type == "top-k":
                    topk_clusters = topk_clusters[:k]
                elif retrieval_type == "similarity":
                    topk_clusters = [(score, cluster) for score, cluster in topk_clusters
                                     if score >= threshold]
                    if not topk_clusters:
                        return torch.zeros_like(target_feature.unsqueeze(0))
            else:
                next_level_clusters = []
                for _, cluster in topk_clusters:
                    parent_ids = cluster["parent_clusters"]
                    if isinstance(parent_ids, list):
                        for parent_id in parent_ids:
                            if f'cluster_{parent_id}' in clusters:
                                sub_cluster = clusters[f'cluster_{parent_id}']
                                sub_score = cosine_similarity(target_feature.unsqueeze(0),
                                                              sub_cluster["clip_embedding"])
                                next_level_clusters.append((sub_score, sub_cluster))
                    else:
                        if f'cluster_{parent_ids}' in clusters:
                            sub_cluster = clusters[f'cluster_{parent_ids}']
                            sub_score = cosine_similarity(target_feature.unsqueeze(0),
                                                          sub_cluster["clip_embedding"])
                            next_level_clusters.append((sub_score, sub_cluster))

                next_level_clusters.sort(key=lambda x: x[0], reverse=True)
                if retrieval_type == "max":
                    next_level_clusters = next_level_clusters[:1]
                elif retrieval_type == "top-k":
                    next_level_clusters = next_level_clusters[:k]
                elif retrieval_type == "similarity":
                    next_level_clusters = [(score, cluster) for score, cluster in next_level_clusters
                                           if score >= threshold]
                    if not next_level_clusters:
                        if combined_vectors:
                            final_embedding = torch.cat(combined_vectors, dim=0).mean(dim=0, keepdim=True)
                            return final_embedding
                        else:
                            return torch.zeros_like(target_feature.unsqueeze(0))

            for _, cluster in topk_clusters:
                if "summary" in cluster:
                    level_summaries[level].append(cluster["summary"])
            if level in selected_levels:
                level_vectors = torch.stack(
                    [cluster["clip_embedding"] for _, cluster in topk_clusters]
                ).squeeze(dim=1)
                if retrieval_type != "max":
                    level_vectors = level_vectors.mean(dim=0, keepdim=True)
                combined_vectors.append(level_vectors)

        final_embedding = torch.cat(combined_vectors, dim=0)
        final_embedding = final_embedding.mean(dim=0, keepdim=True)
        return final_embedding

    def ret(self, target_video, memory_bank, mode, uns_video=None):
        topk_embeds, topk_sents = self.softattention_select(
            memory_bank, target_video, mode, uns_video=uns_video
        )

        if len(topk_embeds) == 0:
            return None

        value_vectors = topk_embeds
        if len(value_vectors.shape) != 3:
            value_vectors = torch.unsqueeze(value_vectors, dim=0)
        b, s, h = value_vectors.shape
        value_vectors = value_vectors.view(b, s, h)

        if self.args.ret_encoder == "avg":
            value_vectors = topk_embeds

        ret = self.ret2t5_proj(value_vectors)
        return ret


# ---------------------------------------------------------------------- #
#  Helper modules (HiCM^2 official)
# ---------------------------------------------------------------------- #
class LinearLayer(nn.Module):
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super().__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz),
        )

    def forward(self, x):
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x
