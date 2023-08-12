"""
The inference results are saved to a csv file.
Usage:
    Inference with model on wandb:
        python inference.py \
        --timm_name {model name in timm} \
        --layer_name {layer name in catalyst} \
        --model_name {model name storaged in wandb} \
        --wandb_run_path {wandb_run_path} \
        --image_size {image size default: 224}
        --embedding_size {embedder output size default: 512} \
        --k {top@k default: 10}
"""
import argparse

import wandb

from src.dataset import get_image_dataset, get_inference_dataloader
from src.model import EncoderWithHead
from src.prediction import InferenceModel, load_weights, predict_fn


def main(args):
    # image dataset
    image_dataset, index2target = get_image_dataset(
        df_dir="input/train.csv",
        img_dir="input/train_data",
        image_size=args.image_size,
    )

    # dataloader
    test_dataloader = get_inference_dataloader(
        df_dir="input/sample_submission.csv",
        img_dir="input/test_data",
        image_size=args.image_size,
    )

    # model
    model = EncoderWithHead(
        model_name=args.timm_name,
        pretrained=False,
        layer_name=args.layer_name,
        embedding_size=args.embedding_size,
        num_classes=args.num_classes,
    )

    # restore model weights in wandb
    best_weights = wandb.restore(
        f"{args.model_name}.ckpt", run_path=args.wandb_run_path
    )

    # load weights
    model = load_weights(
        model=model,
        weights=best_weights,
    )

    # inference
    im = InferenceModel(model.encoder)
    im.train_knn(image_dataset)

    df, df_top1, df_mode = predict_fn(
        inference_model=im,
        test_dataloader=test_dataloader,
        index_to_target=index2target,
        k=args.k,
    )

    df.to_csv(f"submit/inference_{args.model_name}.csv", sep=",", index=None)
    df_top1.to_csv(
        f"submit/submission_top1_{args.model_name}.csv",
        sep=",",
        header=None,
        index=None,
    )
    df_mode.to_csv(
        f"submit/submission_mode_{args.model_name}.csv",
        sep=",",
        header=None,
        index=None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timm_name", type=str)
    parser.add_argument("--layer_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--wandb_run_path", type=str)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=2)

    args = parser.parse_args()

    main(args)
