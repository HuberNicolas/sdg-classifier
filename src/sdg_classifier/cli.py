"""Command-line interface: ``sdg-classifier prepare | train | predict``."""

import argparse
import json
import logging
import sys
from pathlib import Path


def prepare(args: argparse.Namespace) -> None:
    from .data import load_abstracts, one_label_per_row, plot_distribution, split

    df = load_abstracts(args.input)
    print(f"{len(df)} articles with an abstract and at least one SDG")
    if args.one_label_per_row:
        df = one_label_per_row(df)
        print(f"{len(df)} rows after splitting into one label per row")
    dataset = split(df, seed=args.seed)
    dataset.save_to_disk(str(args.output))
    print(dataset)
    if args.plot:
        plot_distribution(df, args.plot)
        print(f"Saved the label distribution to {args.plot}")


def train(args: argparse.Namespace) -> None:
    from .train import train

    metrics = train(
        args.dataset,
        args.output,
        base_model=args.base_model,
        lora=args.lora,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        class_weights=not args.no_class_weights,
        limit=args.limit,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


def predict(args: argparse.Namespace) -> None:
    from .predict import SDGClassifier

    texts = args.text or [line.strip() for line in sys.stdin if line.strip()]
    classifier = SDGClassifier(args.model)
    for text, probs in zip(texts, classifier.probabilities(texts), strict=True):
        sdgs = {sdg: p for sdg, p in probs.items() if p >= args.threshold}
        print(json.dumps({"text": text[:80], "sdgs": sdgs}))


def main(argv: list[str] | None = None) -> None:
    from .train import DEFAULT_BASE_MODEL

    parser = argparse.ArgumentParser(prog="sdg-classifier", description="Multi-label SDG classification of abstracts")
    parser.add_argument("--seed", type=int, default=42)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("prepare", help="clean the abstracts and split them into train/validation/test")
    p.add_argument("--input", type=Path, default=Path("data/sdg_data_abstracts.csv"), help="abstract-scraper output")
    p.add_argument("--output", type=Path, default=Path("data/preprocessed/sdg_dataset_splits_multilabel"))
    p.add_argument("--one-label-per-row", action="store_true", help="repeat articles with several SDGs, once per SDG")
    p.add_argument("--plot", type=Path, help="save a bar chart of the label distribution, e.g. output/distribution.pdf")
    p.set_defaults(func=prepare)

    t = sub.add_parser("train", help="fine-tune the model")
    t.add_argument("--dataset", type=Path, default=Path("data/preprocessed/sdg_dataset_splits_multilabel"))
    t.add_argument("--output", type=Path, default=Path("output/distilbert-sdg-classifier"))
    t.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    t.add_argument("--lora", action="store_true", help="train LoRA adapters instead of all weights")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--learning-rate", type=float, help="default: 2e-5, or 2e-4 with --lora")
    t.add_argument("--max-length", type=int, default=128, help="tokens per abstract")
    t.add_argument("--no-class-weights", action="store_true", help="do not weigh rare SDGs up in the loss")
    t.add_argument("--limit", type=int, help="use only this many examples per split, for a quick test")
    t.set_defaults(func=train)

    r = sub.add_parser("predict", help="print the SDGs of texts given as arguments or on stdin, one per line")
    r.add_argument("text", nargs="*")
    r.add_argument("--model", type=Path, default=Path("output/distilbert-sdg-classifier"))
    r.add_argument("--threshold", type=float, default=0.5)
    r.set_defaults(func=predict)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
