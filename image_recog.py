import argparse
import glob
import os

import numpy as np
import torch
import open_clip
from PIL import Image

DEFAULT_MODELS = [
    ("ViT-B-32", "openai"),
    ("ViT-L-14", "openai"),
]

TARGETS = ["bag", "curtain", "bed", "chair", "bucket", "hat", "clock"]

SYNONYMS = {
    "bag": ["bag", "backpack", "handbag", "tote bag", "suitcase", "briefcase"],
    "curtain": ["curtain", "curtains", "drapes", "window curtain"],
    "bed": ["bed", "bed frame", "mattress"],
    "chair": ["chair", "armchair", "stool", "office chair", "folding chair"],
    "bucket": ["bucket", "pail", "plastic bucket", "metal bucket"],
    "hat": ["hat", "cap", "baseball cap", "beanie", "helmet"],
    "clock": ["clock", "wall clock", "alarm clock", "watch"],
}

POS_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo containing a {}.",
    "an image of a {}.",
    "a picture of a {}.",
    "there is a {} in this image.",
    "a close-up photo of a {}.",
]

BASELINE_CONCEPTS = [
    "traffic light", "fire hydrant", "bicycle", "car", "bus", "train",
    "motorcycle", "boat", "airplane", "bridge", "stairs", "house", "tree",
    "flower", "dog", "cat", "bird", "horse", "cow", "sheep", "person",
    "pizza", "cake", "fruit", "shoe", "umbrella", "lamp", "table", "sofa",
    "television", "laptop", "bottle", "cup", "book", "key", "scissors",
    "glasses", "door", "window", "plant", "truck", "sink", "toilet",
    "bathtub", "mirror", "vase", "candle", "towel", "pillow", "blanket",
    "rug", "painting", "camera", "guitar", "piano", "drum", "trumpet",
    "saxophone", "violin", "soccer ball", "tennis racket", "skateboard",
    "skis", "snowboard", "stop sign", "parking meter", "bench", "mailbox",
    "fountain", "mountains", "beach", "ocean", "forest", "field", "snow",
    "city", "road", "street", "building", "fence", "screen", "text", "logo",
]


class ClipEnsembleDetector:
    def __init__(self, models=None, device=None):
        models = models or DEFAULT_MODELS
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.preprocesses = []
        self.tokenizers = []
        for name, pretrained in models:
            model, _, preprocess = open_clip.create_model_and_transforms(
                name, pretrained=pretrained, device=self.device
            )
            model.eval()
            self.models.append(model)
            self.preprocesses.append(preprocess)
            self.tokenizers.append(open_clip.get_tokenizer(name))
        self._baseline_texts = [
            t.format(c) for c in BASELINE_CONCEPTS for t in ["a photo of a {}."]
        ]
        self._baseline_feats = self._text_features(self._baseline_texts)

    def _text_features(self, texts):
        all_feats = []
        for model, tokenizer in zip(self.models, self.tokenizers):
            tokens = tokenizer(texts).to(self.device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats /= feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
        return all_feats

    def _image_features(self, image_paths):
        all_feats = []
        for model, preprocess in zip(self.models, self.preprocesses):
            imgs = torch.stack(
                [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
            ).to(self.device)
            with torch.no_grad():
                feats = model.encode_image(imgs)
                feats /= feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
        return all_feats

    def score(self, image_paths, target, use_synonyms=True):
        words = SYNONYMS.get(target, [target]) if use_synonyms else [target]
        pos_texts = [t.format(w) for w in words for t in POS_TEMPLATES]
        pos_feats = self._text_features(pos_texts)
        img_feats = self._image_features(image_paths)

        pos_sims, base_sims = [], []
        for pf, bf, imf in zip(pos_feats, self._baseline_feats, img_feats):
            pos_sims.append((imf @ pf.T).max(dim=1).values)
            base_sims.append((imf @ bf.T).mean(dim=1))
        pos_sim = torch.stack(pos_sims).mean(dim=0)
        base_sim = torch.stack(base_sims).mean(dim=0)
        return pos_sim.numpy(), base_sim.numpy()


def detect_group(image_paths, target, detector=None, margin=0.07, top_k=None, use_synonyms=True):
    if detector is None:
        detector = ClipEnsembleDetector()
    pos, base = detector.score(image_paths, target, use_synonyms=use_synonyms)
    diff = pos - base
    verdicts = diff > margin
    if top_k:
        top_idx = np.argsort(diff)[::-1][:top_k]
        verdicts = np.zeros_like(verdicts)
        verdicts[top_idx] = True
    results = []
    for path, p, b, d, v in zip(image_paths, pos, base, diff, verdicts):
        results.append(
            {
                "image": os.path.basename(path),
                "path": path,
                "has_target": bool(v),
                "score": float(d),
                "pos": float(p),
                "baseline": float(b),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Detect which images in a folder contain a given target object."
    )
    parser.add_argument("folder", help="folder with images (or glob pattern)")
    parser.add_argument("target", help="target word, e.g. bag / clock")
    parser.add_argument("--margin", type=float, default=0.07)
    parser.add_argument("--top-k", type=int, default=None, help="force top-k images")
    parser.add_argument("--no-synonyms", action="store_true")
    parser.add_argument("--no-ensemble", action="store_true", help="use only ViT-B-32 for speed")
    args = parser.parse_args()

    if any(ch in args.folder for ch in "*?"):
        image_paths = sorted(glob.glob(args.folder))
    else:
        image_paths = sorted(
            p
            for p in glob.glob(os.path.join(args.folder, "*"))
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
        )
    if not image_paths:
        print("no images found")
        return

    models = [DEFAULT_MODELS[0]] if args.no_ensemble else DEFAULT_MODELS
    detector = ClipEnsembleDetector(models=models)
    results = detect_group(
        image_paths,
        args.target,
        detector,
        args.margin,
        args.top_k,
        use_synonyms=not args.no_synonyms,
    )

    for r in sorted(results, key=lambda x: -x["score"]):
        mark = "YES" if r["has_target"] else "no "
        print(f"{mark}  {r['image']:<16} score={r['score']:+.4f}  (pos={r['pos']:.4f} base={r['baseline']:.4f})")
    yes = [r["image"] for r in results if r["has_target"]]
    print(f"\n{len(yes)}/{len(results)} images contain '{args.target}': {yes}")


if __name__ == "__main__":
    main()
