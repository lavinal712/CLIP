import argparse
import ast


def get_default_params(model_params):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    vision_layers = model_params["vision_layers"]
    if isinstance(vision_layers, (tuple, list)):
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split("=")
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()

    args = parser.parse_args(args)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to directory with data.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./save",
        help="Where to store checkpoints and logs.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        help="LR scheduler. One of: 'cosine', 'const' (constant). Default: cosine",
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
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        nargs="*",
        default={
            "embed_dim": 1024,
            "image_resolution": 224,
            "vision_layers": [3, 4, 6, 3],
            "vision_width": 64,
            "vision_patch_size": 16,
            "context_length": 77,
            "vocab_size": 49408,
            "transformer_width": 512,
            "transformer_heads": 8,
            "transformer_layers": 12,
        },
        action=ParseKwargs,
        help="Parameters of the model to use.",
    )
    parser.add_argument(
        "--image-mean", type=float, nargs="+", default=(0.48145466, 0.4578275, 0.40821073), metavar="MEAN",
        help="Override default image mean value of dataset"
    )
    parser.add_argument(
        "--image-std", type=float, nargs="+", default=(0.26862954, 0.26130258, 0.27577711), metavar="STD",
        help="Override default image std deviation of of dataset"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )

    args = parser.parse_args(args)

    # set default opt params based on model name (only if timm optimizer not used)
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)
    

    return args
