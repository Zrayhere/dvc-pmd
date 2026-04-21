import argparse
import os

DATA_DIR = "./data"
PRESAVE_DIR = "./presave"
MODEL_DIR = "./presave"

name2folder = {
    "youcook": "yc2",
    "vitt": "vitt",
}


def get_args_parser():
    parser = argparse.ArgumentParser("Set PMD", add_help=False)

    # ===== Retrieval branch (HiCM^2) =====
    parser.add_argument("--ret_path", type=str, default=False, help='memory path, if True, load directly.')
    parser.add_argument("--hier_level", type=int, default=0)
    parser.add_argument("--LLM_ver", type=int, default=8)
    parser.add_argument("--hier_ret_num", type=str, default='top-k', help='max,top-k,adaptive')
    parser.add_argument("--hier_use", nargs="+", help="1~n")

    parser.add_argument("--ret_encoder", type=str, default='avg', help="avg,miniTE,top1,attention")
    parser.add_argument("--sampling", type=str, default='origin', help="origin,average,max")
    parser.add_argument("--ret_option", type=str, default='hier_concat',
                        help="hier_concat (HiCM^2 retrieval), no_ret (Vid2Seq backbone)")
    parser.add_argument("--sim_match", type=str, default='anchor_cos', help="anchor_cos,attn,multi_attn,...")
    parser.add_argument("--soft_k", type=int, default=10)
    parser.add_argument("--ret2t5_proj", type=str, default='deep', help="simple,deep")
    parser.add_argument("--bank_path", type=str, default="./data/bank")
    parser.add_argument("--bank_type", nargs='+', default=['anet'],
                        help="domains used in retrieval bank")
    parser.add_argument("--window_size", type=int, default=10, help="window number")
    parser.add_argument("--drop_last_enable", type=bool, default=False)

    # ===== Datasets =====
    parser.add_argument(
        "--combine_datasets",
        nargs="+",
        help="list of datasets to combine for training",
        required=True,
    )
    parser.add_argument(
        "--combine_datasets_val",
        nargs="+",
        help="list of datasets to combine for eval",
        default=[],
    )

    # YouCook2
    parser.add_argument(
        "--youcook_features_path",
        default=os.path.join(DATA_DIR, name2folder["youcook"], "clipvitl14.pth"),
    )
    parser.add_argument(
        "--youcook_train_json_path",
        default=os.path.join(DATA_DIR, name2folder["youcook"], "train.json"),
    )
    parser.add_argument(
        "--youcook_val_json_path",
        default=os.path.join(DATA_DIR, name2folder["youcook"], "val.json"),
    )
    parser.add_argument(
        "--youcook_subtitles_path",
        default=os.path.join(DATA_DIR, name2folder["youcook"], "youcook2_asr_align_proc.pkl"),
    )

    # ViTT
    parser.add_argument(
        "--vitt_features_path",
        default=os.path.join(DATA_DIR, name2folder["vitt"], "clipvitl14.pth"),
    )
    parser.add_argument(
        "--vitt_train_json_path",
        default=os.path.join(DATA_DIR, name2folder["vitt"], "train.json"),
    )
    parser.add_argument(
        "--vitt_val_json_path",
        default=os.path.join(DATA_DIR, name2folder["vitt"], "dev.json"),
    )
    parser.add_argument(
        "--vitt_test_json_path",
        default=os.path.join(DATA_DIR, name2folder["vitt"], "test.json"),
    )
    parser.add_argument(
        "--vitt_subtitles_path",
        default=os.path.join(DATA_DIR, name2folder["vitt"], "subtitles_align_proc.pkl"),
    )

    # ===== Training hyper-parameters =====
    parser.add_argument("--denoising", default=1., type=float, help="denoising loss coef")
    parser.add_argument("--generative", default=1., type=float, help="generative loss coef")
    parser.add_argument("--mask_prob", type=float, default=0.25,
                        help="masking probability for the denoising objective")
    parser.add_argument("--mask_len", type=int, default=5,
                        help="masking average span length for the denoising objective")
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate")
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--batch_size_val", default=2, type=int)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--clip_max_norm", default=1., type=float, help="gradient clipping max norm")
    parser.add_argument("--schedule", default="",
                        choices=["", "cosine_with_warmup"],
                        help="learning rate decay schedule, default is constant")
    parser.add_argument("--fraction_warmup_steps", default=0.1, type=float)
    parser.add_argument("--eval_skip", default=3, type=int)
    parser.add_argument("--eval_skip2", default=5, type=int)
    parser.add_argument("--print_freq", type=int, default=100)

    # ===== Run specific =====
    parser.add_argument("--save_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--presave_dir", default=PRESAVE_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--load", default="", help="path to load checkpoint")
    parser.add_argument("--resume", action="store_true",
                        help="continue training if loading checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N")
    parser.add_argument("--eval", action="store_true", help="only run evaluation")
    parser.add_argument("--num_workers", default=3, type=int)

    # ===== Distributed training parameters =====
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--dist-url", default="env://")

    # ===== Model parameters =====
    parser.add_argument("--model_name", default="t5-base", choices=("t5-base",))
    parser.add_argument("--text_encoder_dropout", default=0.1, type=float)
    parser.add_argument("--text_decoder_dropout", default=0.1, type=float)
    parser.add_argument("--visual_encoder_dropout", default=0.1, type=float)
    parser.add_argument("--max_feats", type=int, default=100,
                        help="maximum number of video features considered, one per frame")
    parser.add_argument("--features_dim", type=int, default=768,
                        help="dimension of the visual embedding space")
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--mlp_dim", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--num_bins", type=int, default=100,
                        help="number of quantization bins for the time tokens")
    parser.add_argument("--no_video", dest="use_video", action="store_false",
                        help="disables usage of video")
    parser.add_argument("--no_speech", dest="use_speech", action="store_false",
                        help="disables usage of speech")
    parser.add_argument("--max_input_tokens", type=int, default=1000,
                        help="maximum number of tokens in the input speech")
    parser.add_argument("--max_output_tokens", type=int, default=256,
                        help="maximum number of tokens in the output sequence of dense captions")

    # ===== Generation =====
    parser.add_argument("--num_beams", type=int, default=4, help="beam search size")
    parser.add_argument("--length_penalty", type=float, default=1.)
    parser.add_argument("--repetition_penalty", type=float, default=1.)
    parser.add_argument("--top_p", type=float, default=0.9)

    # =====================================================================
    # PMD COMPONENTS
    # =====================================================================

    # ----- Phase 2: SaliGT (geometric boundary grounding) -----
    # Apply frame-level sigmoid weighting around GT event boundaries during
    # training; full visual signal is used at inference.
    parser.add_argument('--use_saliency_reweight', action='store_true', default=False,
                        help='Phase 2 SaliGT: GT-driven saliency reweighting at training time')
    parser.add_argument('--saliency_alpha', type=float, default=10.0,
                        help='Sigmoid sharpness for SaliGT')

    # ----- Phase 3: VMA (visual modality attenuation) -----
    # Note: in code this is referred to as "VMA" (the original variable name);
    # the paper calls it VMA. When --use_vma is set, visual features are scaled
    # by --vma_lambda during training; the scaling block is bypassed at
    # inference (gated by self.training), so the full visual signal is always
    # used at evaluation time.
    #
    # The paper sets vma_lambda=0 throughout Phase 3 (full attenuation). The
    # lambda sweep ablation (Tab. V in the paper) uses values in {0, 0.3, 0.5, 1.0}.
    parser.add_argument('--use_vma', action='store_true', default=False,
                        help='VMA (Visual Modality Attenuation): training-time visual scaling')
    parser.add_argument('--vma_lambda', type=float, default=0.0,
                        help='Visual scale lambda during training (paper default: 0.0)')

    # ----- Phase 3: STT (semantic transition tokens) -----
    # Inject pre-extracted boundary tokens through the ASR text channel.
    # Tokens are produced offline by the SSTA pipeline (see scripts/build_stt.py).
    parser.add_argument('--use_boundary_tokens', action='store_true', default=False,
                        help='STT: append structured boundary tokens to ASR input during training')
    parser.add_argument('--boundary_tokens_path', type=str,
                        default='./data/yc2/boundary_tokens.json',
                        help='Path to precomputed STT boundary tokens JSON')
    parser.add_argument('--filter_weak_boundary', action='store_true', default=False,
                        help='STT: skip weak boundaries during training (recommended)')

    return parser
