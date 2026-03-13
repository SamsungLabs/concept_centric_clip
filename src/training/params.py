import argparse
import ast


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto", "npy", "json", "cc3m_custom", "cc30m_custom", "cc30m_custom_hn_np"],
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        '--image-interpolation',
        default=None, type=str, choices=['bicubic', 'bilinear', 'random'],
        help="Override default image resize interpolation"
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None, type=str, choices=['shortest', 'longest', 'squash'],
        help="Override default image resize (& crop) mode during inference"
    )
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="unsupervised-detection"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze LayerNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa."
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa."
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help='Use SigLip (sigmoid) loss.'
    )

    # new params
    parser.add_argument(
        "--images-dir-path",
        type=str,
        default=None,
        help='Path to images directory. Required when using JSON file of BLIP2 generated captions.'
    )
    parser.add_argument(
        "--num-captions",
        type=int,
        default=1,
        help='Number of captions per image. Used with BLIP2 generated captions.'
    )
    parser.add_argument(
        "--use-pseudo-labels",
        action="store_true",
        help="Uses text-text similarities to create pseudo labels for training."
    )
    parser.add_argument(
        "--pseudo-txt-txt-th",
        type=float,
        default=0.99,
        help="Threshold for text-text similarity for pseudo labels."
    )
    parser.add_argument(
        "--captions-key",
        type=str,
        default="blip2_captions",
        help='The key of the captions/tags in the generated/processed JSON file.'
    )

    ############# NEW ###############
    parser.add_argument(
        "--cc3m-captions",
        type=str,
        default=None,
        help="Path to pickled CC3m caption with hard negatives",
    )
    parser.add_argument(
        "--hardnegative",
        default=False,
        action="store_true",
        help="Whether to use da examples to compute clip loss"
    )
    parser.add_argument(
        "--threshold-type",
        choices=["mean", "max","fixed"],
        default="mean",
        help="how to compute threshold"
    )
    parser.add_argument(
        "--fixed-threshold-value",
        type=float,
        default=2.0,
        help="fixed threshold value"
    )
    parser.add_argument(
        "--upper-bound",
        type=int,
        default=10,
        help="clamp upper bound for threshold"
    )
    parser.add_argument(
        "-imc",
        "--imc-loss",
        default=False,
        action="store_true",
        help="whether to use imc loss"
    )
    parser.add_argument(
        "--imc-loss-weight",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "-cmr",
        "--cmr-loss",
        default=False,
        action="store_true",
        help="Whether to use cmr loss"
    )
    parser.add_argument(
        "--cmr-loss-weight",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--output-tokens",
        default=False,
        action="store_true",
        help="flag the vision and text encoders to return token features"
    )
    parser.add_argument(
        "--force-siglip-preprocess", action="store_true", default=False,
    )
    ########## SCAN loss ##########
    parser.add_argument("--scan-loss", default=False, action="store_true", help="Use SCAN loss")
    parser.add_argument("--scan-loss-weight", type=float, default=1.)
    parser.add_argument("--scan-feature-norm", type=str, choices=["clipped_l2norm", "softmax", "l2norm", "l1norm", "clipped_l1norm", "clipped", "no_norm"], default="clipped_l2norm")
    parser.add_argument("--scan-agg-func", type=str, choices=["LogSumExp", "LogSumExp_norm", "Max", "Sum", "Mean"], default="LogSumExp_norm")
    parser.add_argument("--scan-lambda-lse", type=float, default=6.)
    parser.add_argument("--scan-lambda-softmax", type=float, default=9.)
    parser.add_argument("--scan-loss-type", type=str, default="sigmoid", choices=["sigmoid", "hinge"])
    parser.add_argument("--scan-margin", type=float, default=0.2)
    parser.add_argument("--scan-hard-negative", default=False, action="store_true")
    parser.add_argument("--scan-ce-hard-negative", default=False, action="store_true")

    ########## Noun-phrase loss ##########
    parser.add_argument("--np-loss", default=False, action="store_true", help="enable Noun-phrase loss - including 3 sub-objectives")
    parser.add_argument("--without-instance-np-loss", dest="np_instance_loss", default=True, action="store_false", help="disable noun-phrase instance level loss")
    parser.add_argument("--without-token-np-loss", dest="np_token_loss", default=True, action="store_false", help="disable noun-phrase token level loss")
    parser.add_argument("--without-np-intramodal-loss", dest="np_intramodal", default=True, action="store_false", help="disable intromodal noun-phrase loss")
    parser.add_argument("--np-intramodal-loss-scale", type=float, default=0.01)
    parser.add_argument("--np-token-loss-scale", type=float, default=1)
    parser.add_argument("--np-loss-weight", type=float, default=1., help="weight of Noun-phrase loss")
    parser.add_argument("--np-token-token-loss", dest="np_token_token_loss", default=False, action="store_true")
    parser.add_argument("--np-token-token-loss-scale", type=float, default=1.)
    parser.add_argument("--np-flair-loss", dest="np_flair_loss", default=False, action="store_true")
    parser.add_argument("--np-flair-loss-scale", type=float, default=1.)
    parser.add_argument("--np-hard-negative-loss", dest="np_hard_negative_loss", default=False, action="store_true")
    parser.add_argument("--np-hard-negative-loss-scale", type=float, default=1.)
    parser.add_argument("--np-hard-negative-flair-loss", dest="np_hard_negative_flair_loss", default=False, action="store_true")
    parser.add_argument("--np-hard-negative-flair-loss-scale", type=float, default=1.)
    parser.add_argument("--hn-np-balance", default=False, action="store_true")


    ########## FLAIR loss ##########
    parser.add_argument("--flair-loss", default=False, action="store_true", help="enable Noun-phrase loss - including 3 sub-objectives")
    parser.add_argument("--flair-loss-scale", type=float, default=1.)

    ################################
    parser.add_argument("--loss-feature-norm", default=False, action="store_true", help="normalize features in loss function instead")

    parser.add_argument("--use-original-caption", default=False, action="store_true", help="use original cc3m caption")

    ################## END NEW ###############

    args = parser.parse_args(args)

    if args.dataset_type == "cc3m_custom":
        assert args.cc3m_captions is not None and args.train_data is not None, "Must provide both WDS root dir and Augmented caption pickle"

    if args.dataset_type == "json":
        assert args.images_dir_path is not None, "Must specify --image-dir-path if using JSON file of BLIP2 generated captions."

    if args.distill_model is not None:
        assert args.distill_pretrained is not None, "Must specify --distill-pretrained if --distill-model is set"

    if args.use_pseudo_labels:
        assert args.distill_model is not None and args.distill_pretrained is not None, \
            "Must specify --distill-model and --distill-pretrained if --use-pseudo-labels"

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)
    ####### new ######
    if args.hardnegative or args.imc_loss or args.cmr_loss or args.scan_ce_hard_negative:
        print("using extra data-augmented data to train the model")
        args.extra_da=True
    else:
        args.extra_da=False

    # if args.accum_freq != 1 and args.extra_da:
    #     raise ValueError("Currently CE-CLIP does not support gradient accumulation")

    if not args.np_loss:
        args.np_instance_loss = False
        args.np_token_loss = False
        args.np_intramodal = False
    
    if (args.scan_loss or args.np_flair_loss or args.np_hard_negative_flair_loss or args.flair_loss or args.np_token_loss or args.np_token_token_loss) and not args.output_tokens:
        raise ValueError("If SCAN loss or FLAIR loss is used, must also enable [output_tokens] option")
    
    if args.np_loss and (not args.np_instance_loss) and (not args.np_token_loss) and (not args.np_intramodal) and (not args.np_token_token_loss) and (not args.np_flair_loss) and (not args.np_hard_negative_flair_loss):
        raise ValueError("At least one option of NP loss must be enabled")
    
    # if args.np_loss and (args.np_token_loss or args.np_token_token_loss or args.np_flair_loss or args.np_flair_loss or args.np_hard_negative_flair_loss):
    #     args.output_tokens = True

    ##################
    return args