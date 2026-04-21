import os
import json
import pickle
import numpy as np
import torch as th
import torch
from torch.utils.data import Dataset

from util.t5 import create_sentinel_ids, filter_input_ids, random_spans_noise_mask


class DenseVideoCaptioning_Dataset(Dataset):
    """
    Dense Video Captioning dataset for HiCM^2 backbone.

    PMD-related fields (all optional, gated by args flags):
      - saliency_weights         : Phase 2 SaliGT frame-level reweighting (use_saliency_reweight)
      - input_tokens (augmented) : Phase 3 STT boundary tokens appended to ASR
                                   (use_boundary_tokens)
    """

    def __init__(
        self,
        json_path,
        features_path,
        max_feats=100,
        features_dim=768,
        tokenizer=None,
        subtitles_path=None,
        num_bins=100,
        max_input_tokens=1000,
        max_output_tokens=256,
        noise_density=0.25,
        mean_noise_span_length=5,
        dataset_name=None,
        args=None,
    ):
        self.data = json.load(open(json_path, 'r'))
        self.vids = list(self.data.keys())
        self.features = None
        self.features_path = None
        self.dataset_name = dataset_name
        self.args = args

        # ---------------- PMD switches ----------------
        # Phase 2 SaliGT: frame-level sigmoid reweighting around GT boundaries
        self.use_saliency = getattr(args, 'use_saliency_reweight', False) if args is not None else False
        self.saliency_alpha = getattr(args, 'saliency_alpha', 10.0) if args is not None else 10.0

        # Phase 3 STT: load pre-extracted boundary tokens
        self.use_boundary_tokens = (
            getattr(args, 'use_boundary_tokens', False) if args is not None else False
        )
        self.boundary_tokens_data = None
        if self.use_boundary_tokens:
            bt_path = getattr(args, 'boundary_tokens_path', None)
            if bt_path and os.path.exists(bt_path):
                self.boundary_tokens_data = json.load(open(bt_path, 'r'))
                print(f"[STT] Loaded boundary tokens from {bt_path}, "
                      f"{len(self.boundary_tokens_data)} videos")
            else:
                print(f"[STT] WARNING: boundary_tokens_path not found: {bt_path}, disabling")
                self.use_boundary_tokens = False
        # ----------------------------------------------

        # Load CLIP features (pickle/.pth dict mapping video_id -> [T, d] tensor)
        if os.path.isdir(features_path):
            self.features_path = features_path
        else:
            self.features = th.load(features_path)

        self.max_feats = max_feats
        self.features_dim = features_dim
        self.tokenizer = tokenizer

        # Load ASR subtitles
        self.subs = None
        self.subs_path = None
        if subtitles_path and os.path.exists(subtitles_path) and os.path.isdir(subtitles_path):
            self.subs_path = subtitles_path
        elif subtitles_path and os.path.exists(subtitles_path):
            self.subs = pickle.load(open(subtitles_path, "rb"))
        else:
            print("No subtitles given or found.")

        self.num_bins = num_bins
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.num_text_tokens = len(tokenizer) - num_bins
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length

    def __len__(self):
        return len(self.data)

    def _get_text(self, text):
        text = text.strip()
        text = text.capitalize()
        if text[-1] != '.':
            text = text + '.'
        return text

    def _get_video(self, video_id):
        if self.features is not None:
            assert video_id in self.features, video_id
            video = self.features[video_id].float()
        else:
            features_path = os.path.join(self.features_path, video_id + '.mp4.npy')
            if not os.path.exists(features_path):
                features_path = os.path.join(self.features_path, video_id + '.npy')
            assert os.path.exists(features_path), features_path
            video = th.from_numpy(np.load(features_path)).float()

        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = th.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = th.cat(
                [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
            )
        else:
            video_len = self.max_feats

        return video, video_len

    def time_tokenize(self, x, duration, num_bins):
        time_token = int(float((num_bins - 1) * x) / float(duration))
        assert time_token <= self.num_bins
        return time_token + self.num_text_tokens

    def __getitem__(self, idx):
        video_id = self.vids[idx]
        annotations = self.data[video_id]
        video, video_len = self._get_video(video_id[-11:])
        duration = annotations["duration"]

        # ---------------- ASR input tokens ----------------
        if (self.subs is not None and video_id[-11:] in self.subs) or (
                self.subs_path is not None and os.path.exists(os.path.join(self.subs_path, video_id + '.pkl'))):
            if self.subs is not None and video_id[-11:] in self.subs:
                sub = self.subs[video_id[-11:]]
            else:
                sub = pickle.load(open(os.path.join(self.subs_path, video_id[-11:] + '.pkl'), 'rb'))
            to_keep = [(x >= 0 and y <= duration) for x, y in zip(sub["start"], sub["end"])]
            if not any(to_keep):
                input_tokens = (th.ones(1) * self.tokenizer.eos_token_id).long()
            else:
                sub["start"] = [x for i, x in enumerate(sub["start"]) if to_keep[i]]
                sub["end"] = [x for i, x in enumerate(sub["end"]) if to_keep[i]]
                sub['text'] = [self._get_text(x) for i, x in enumerate(sub['text']) if to_keep[i]]
                time_input_tokens = [
                    th.LongTensor([self.time_tokenize(st, duration, self.num_bins),
                                   self.time_tokenize(ed, duration, self.num_bins)])
                    for st, ed in zip(sub['start'], sub['end'])
                ]
                text_input_tokens = [
                    self.tokenizer(x, add_special_tokens=False, max_length=self.max_input_tokens,
                                   padding="do_not_pad", truncation=True, return_tensors="pt")['input_ids'][0]
                    for x in sub['text']
                ]
                input_tokens = [th.cat([ti, te], 0) for ti, te in zip(time_input_tokens, text_input_tokens)]
                input_tokens = th.cat(input_tokens, 0)
                input_tokens = input_tokens[:self.max_input_tokens - 1]
                input_tokens = th.cat([input_tokens, th.LongTensor([self.tokenizer.eos_token_id])], 0)
        else:
            input_tokens = (th.ones(1) * self.tokenizer.eos_token_id).long()

        # ---------------- Denoising sequence (HiCM^2 baseline) ----------------
        if len(input_tokens) > 1:
            mask_indices = np.asarray(
                [random_spans_noise_mask(len(input_tokens), self.noise_density, self.mean_noise_span_length)])
            labels_mask = ~mask_indices

            input_ids_sentinel = create_sentinel_ids(mask_indices.astype(np.int8), self.tokenizer, self.num_bins)
            labels_sentinel = create_sentinel_ids(labels_mask.astype(np.int8), self.tokenizer, self.num_bins)

            denoising_output_tokens = th.from_numpy(
                filter_input_ids(input_tokens.unsqueeze(0).numpy(), labels_sentinel, self.tokenizer)).squeeze(0)
            denoising_input_tokens = th.from_numpy(
                filter_input_ids(input_tokens.unsqueeze(0).numpy(), input_ids_sentinel, self.tokenizer)).squeeze(0)
        else:
            input_tokens = th.LongTensor([self.tokenizer.eos_token_id])
            denoising_input_tokens = th.LongTensor([0])
            denoising_output_tokens = input_tokens

        # ============================================================
        # PHASE 3 STT: append structured boundary tokens to ASR input
        # ============================================================
        # Each boundary entry is a typed transition phrase wrapped between two
        # Vid2Seq time-bin tokens. Weak boundaries are skipped (paper Sec. III-C).
        if self.use_boundary_tokens and self.boundary_tokens_data is not None:
            bt_data = self.boundary_tokens_data.get(video_id, None)
            if bt_data is not None and bt_data.get("n_boundaries", 0) > 0:
                boundary_parts = bt_data["boundary_parts"]
                boundary_text_parts = []
                for bp, detail in zip(boundary_parts, bt_data.get("details", [])):
                    if getattr(self.args, 'filter_weak_boundary', False) and \
                            detail.get("level", "weak") == "weak":
                        continue
                    t_end = bp["t_end_prev"]
                    t_start = bp["t_start_next"]
                    desc = bp["text"]

                    time_token_end = self.time_tokenize(t_end, duration, self.num_bins)
                    time_token_start = self.time_tokenize(t_start, duration, self.num_bins)

                    desc_tokens = self.tokenizer(
                        desc,
                        add_special_tokens=False,
                        max_length=20,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors="pt",
                    )['input_ids'][0]

                    boundary_tokens = th.cat([
                        th.LongTensor([time_token_end]),
                        desc_tokens,
                        th.LongTensor([time_token_start]),
                    ])
                    boundary_text_parts.append(boundary_tokens)

                if boundary_text_parts:
                    all_boundary_tokens = th.cat(boundary_text_parts)
                    remaining = self.max_input_tokens - len(input_tokens) - 1
                    if remaining > 0:
                        all_boundary_tokens = all_boundary_tokens[:remaining]
                        if input_tokens[-1] == self.tokenizer.eos_token_id:
                            input_tokens = th.cat([
                                input_tokens[:-1],
                                all_boundary_tokens,
                                th.LongTensor([self.tokenizer.eos_token_id]),
                            ])
                        else:
                            input_tokens = th.cat([input_tokens, all_boundary_tokens])

        # ---------------- DVC output sequence ----------------
        captions = [self._get_text(x) for x in annotations['sentences']]
        time_output_tokens = [
            th.LongTensor([self.time_tokenize(st, duration, self.num_bins),
                           self.time_tokenize(ed, duration, self.num_bins)])
            for st, ed in annotations['timestamps']
        ]
        text_output_tokens = [
            self.tokenizer(x, add_special_tokens=False, max_length=self.max_output_tokens,
                           padding="do_not_pad", truncation=True, return_tensors="pt")['input_ids'][0]
            for x in captions
        ]
        output_tokens = [th.cat([ti, te], 0) for ti, te in zip(time_output_tokens, text_output_tokens)]
        output_tokens = th.cat(output_tokens, 0)
        output_tokens = output_tokens[:self.max_output_tokens - 1]
        output_tokens = th.cat([output_tokens, th.LongTensor([self.tokenizer.eos_token_id])], 0)

        result = {
            "video_id": video_id,
            "duration": duration,
            "video": video,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "denoising_input_tokens": denoising_input_tokens,
            "denoising_output_tokens": denoising_output_tokens,
        }

        # ============================================================
        # PHASE 2 SaliGT: frame-level sigmoid reweighting (paper Sec. III)
        # w(i; E) = max_n g(i/T, t_s^n / d, t_e^n / d) with sharpness alpha
        # ============================================================
        if self.use_saliency:
            alpha_base = self.saliency_alpha  # 10.0 per paper

            T = self.max_feats
            timestamps = annotations["timestamps"]
            frame_pos = th.arange(T).float() / T  # normalized [0, 1)

            W = th.zeros(T)
            for t_s, t_e in timestamps:
                x_l = alpha_base * (frame_pos - t_s / duration)
                x_r = alpha_base * (t_e / duration - frame_pos)
                w_l = th.sigmoid(x_l)
                w_r = th.sigmoid(x_r)
                w_event = w_l * w_r
                W = th.max(W, w_event)
            result["saliency_weights"] = W

        return result


def densevideocaptioning_collate_fn(batch):
    bs = len(batch)
    video_id = [batch[i]["video_id"] for i in range(bs)]
    duration = [batch[i]["duration"] for i in range(bs)]
    video = th.stack([batch[i]["video"] for i in range(bs)])

    input_tokens = [batch[i]["input_tokens"] for i in range(bs)]
    max_input_len = max(len(x) for x in input_tokens)
    for i in range(bs):
        if len(input_tokens[i]) < max_input_len:
            input_tokens[i] = th.cat(
                [input_tokens[i], th.zeros(max_input_len - len(input_tokens[i])).long()], 0)
    input_tokens = th.stack(input_tokens)

    output_tokens = [batch[i]["output_tokens"] for i in range(bs)]
    max_output_len = max(len(x) for x in output_tokens)
    for i in range(bs):
        if len(output_tokens[i]) < max_output_len:
            output_tokens[i] = th.cat(
                [output_tokens[i], th.zeros(max_output_len - len(output_tokens[i])).long()], 0)
    output_tokens = th.stack(output_tokens)

    denoising_input_tokens = [batch[i]["denoising_input_tokens"] for i in range(bs)]
    max_input_len = max(len(x) for x in denoising_input_tokens)
    for i in range(bs):
        if len(denoising_input_tokens[i]) < max_input_len:
            denoising_input_tokens[i] = th.cat(
                [denoising_input_tokens[i],
                 th.zeros(max_input_len - len(denoising_input_tokens[i])).long()], 0)
    denoising_input_tokens = th.stack(denoising_input_tokens)

    denoising_output_tokens = [batch[i]["denoising_output_tokens"] for i in range(bs)]
    max_denoising_output_len = max(len(x) for x in denoising_output_tokens)
    for i in range(bs):
        if len(denoising_output_tokens[i]) < max_denoising_output_len:
            denoising_output_tokens[i] = th.cat(
                [denoising_output_tokens[i],
                 th.zeros(max_denoising_output_len - len(denoising_output_tokens[i])).long()], 0)
    denoising_output_tokens = th.stack(denoising_output_tokens)

    out = {
        "video_id": video_id,
        "duration": duration,
        "video": video,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "denoising_input_tokens": denoising_input_tokens,
        "denoising_output_tokens": denoising_output_tokens,
    }

    # Optional PMD fields, only collated if produced by the dataset
    if "saliency_weights" in batch[0]:
        out["saliency_weights"] = th.stack([batch[i]["saliency_weights"] for i in range(bs)])

    return out


def build_densevideocaptioning_dataset(dataset_name, split, args, tokenizer):
    if dataset_name == "youcook":
        if split == "train":
            json_path = args.youcook_train_json_path
        elif split == "val":
            json_path = args.youcook_val_json_path
        else:
            raise NotImplementedError
        features_path = args.youcook_features_path
        subtitles_path = args.youcook_subtitles_path
    elif dataset_name == "vitt":
        if split == "train":
            json_path = args.vitt_train_json_path
        elif split == "val":
            json_path = args.vitt_val_json_path
        elif split == "test":
            json_path = args.vitt_test_json_path
        else:
            raise NotImplementedError
        features_path = args.vitt_features_path
        subtitles_path = args.vitt_subtitles_path
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")

    return DenseVideoCaptioning_Dataset(
        json_path=json_path,
        features_path=features_path,
        max_feats=args.max_feats,
        features_dim=args.features_dim,
        tokenizer=tokenizer,
        subtitles_path=subtitles_path,
        num_bins=args.num_bins,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        dataset_name=dataset_name,
        args=args,
    )