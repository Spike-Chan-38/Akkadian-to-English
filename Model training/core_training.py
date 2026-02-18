import argparse
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

SOURCE_COLUMN_CANDIDATES = ["sentence_transliteration", "source", "akkadian", "transliteration", "text"]
TARGET_COLUMN_CANDIDATES = ["sentence_translation", "target", "english", "translation", "label"]
KEY_COLUMN_CANDIDATES = ["oare_id", "id", "sentence_id"]


# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    src_vocab_size: int = 12000
    tgt_vocab_size: int = 12000
    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 256

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3

    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    epochs: int = 15
    grad_clip: float = 1.0

    val_ratio: float = 0.1
    seed: int = 42


# -------------------------
# Data
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[`]", "'", text)
    return text.strip()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir / path_str
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find file: {path_str}. Tried: '{path}' and '{candidate}'.")


def _pick_first_existing(columns: List[str], candidates: List[str]) -> str:
    for name in candidates:
        if name in columns:
            return name
    return ""


def load_parallel_data(train_csv: str, sentences_csv: str = "") -> pd.DataFrame:
    train_path = _resolve_path(train_csv)
    train_df = pd.read_csv(train_path)

    # Preferred path: train.csv already contains source/target pairs.
    src_col = _pick_first_existing(list(train_df.columns), SOURCE_COLUMN_CANDIDATES)
    tgt_col = _pick_first_existing(list(train_df.columns), TARGET_COLUMN_CANDIDATES)

    if src_col and tgt_col:
        sentence_data = train_df[[src_col, tgt_col]].dropna()
    else:
        if not sentences_csv:
            raise ValueError(
                "Could not find source/target columns in train CSV and no supplementary "
                "--sentences_csv was provided.\n"
                f"train columns: {list(train_df.columns)}"
            )

        sentences_path = _resolve_path(sentences_csv)
        sentences_df = pd.read_csv(sentences_path)
        sent_src_col = _pick_first_existing(list(sentences_df.columns), SOURCE_COLUMN_CANDIDATES)
        sent_tgt_col = _pick_first_existing(list(sentences_df.columns), TARGET_COLUMN_CANDIDATES)
        sent_key_col = _pick_first_existing(list(sentences_df.columns), KEY_COLUMN_CANDIDATES)
        train_key_col = _pick_first_existing(list(train_df.columns), KEY_COLUMN_CANDIDATES)

        if sent_src_col and sent_tgt_col:
            sentence_data = sentences_df[[sent_src_col, sent_tgt_col]].dropna()
            src_col, tgt_col = sent_src_col, sent_tgt_col
        elif sent_key_col and train_key_col and sent_key_col == train_key_col:
            data = pd.merge(sentences_df, train_df, on=sent_key_col, how="left")
            src_col = _pick_first_existing(list(data.columns), SOURCE_COLUMN_CANDIDATES)
            tgt_col = _pick_first_existing(list(data.columns), TARGET_COLUMN_CANDIDATES)
            if not src_col or not tgt_col:
                raise ValueError(
                    "Could not find source/target columns after merge. "
                    f"Available columns: {list(data.columns)}"
                )
            sentence_data = data[[src_col, tgt_col]].dropna()
        else:
            raise ValueError(
                "Could not build parallel data. "
                "Need either source/target columns directly in train/sentences CSV, "
                "or a shared key (e.g., oare_id) in both CSVs.\n"
                f"train columns: {list(train_df.columns)}\n"
                f"sentences columns: {list(sentences_df.columns)}"
            )

    sentence_data = sentence_data.rename(columns={src_col: "source", tgt_col: "target"})

    sentence_data["source"] = sentence_data["source"].apply(normalize_text)
    sentence_data["target"] = sentence_data["target"].apply(normalize_text)

    sentence_data = sentence_data[(sentence_data["source"].str.len() > 0) & (sentence_data["target"].str.len() > 0)]
    sentence_data = sentence_data.drop_duplicates().reset_index(drop=True)
    return sentence_data


def load_test_data(train_csv: str, test_csv: str = "") -> Tuple[pd.DataFrame, str]:
    if test_csv:
        test_path = _resolve_path(test_csv)
    else:
        train_path = _resolve_path(train_csv)
        candidate = train_path.parent / "test.csv"
        if not candidate.exists():
            raise FileNotFoundError(
                "Could not infer test.csv location from train.csv directory. "
                "Pass --test_csv explicitly."
            )
        test_path = candidate

    test_df = pd.read_csv(test_path)
    src_col = _pick_first_existing(list(test_df.columns), SOURCE_COLUMN_CANDIDATES)
    if not src_col:
        raise ValueError(
            "Could not find source column in test CSV. "
            f"Available columns: {list(test_df.columns)}"
        )

    out_df = test_df.copy()
    out_df["source"] = out_df[src_col].astype(str).apply(normalize_text)
    out_df = out_df[out_df["source"].str.len() > 0].reset_index(drop=True)
    return out_df, src_col


def split_train_val(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = list(range(len(df)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_size = int(len(idx) * val_ratio)
    val_idx = set(idx[:val_size])
    train_rows, val_rows = [], []

    for i, row in df.iterrows():
        if i in val_idx:
            val_rows.append(row)
        else:
            train_rows.append(row)

    train_df = pd.DataFrame(train_rows).reset_index(drop=True)
    val_df = pd.DataFrame(val_rows).reset_index(drop=True)
    return train_df, val_df


# -------------------------
# Tokenizers (SentencePiece)
# -------------------------
def _write_text_file(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def train_sentencepiece(texts: List[str], model_prefix: str, vocab_size: int, cfg: Config) -> str:
    txt_path = Path(model_prefix + ".txt")
    _write_text_file(txt_path, texts)

    spm.SentencePieceTrainer.Train(
        input=str(txt_path),
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=1.0,
        pad_id=cfg.pad_id,
        bos_id=cfg.bos_id,
        eos_id=cfg.eos_id,
        unk_id=cfg.unk_id,
        input_sentence_size=200000,
        shuffle_input_sentence=True,
        # Important for low-resource corpora: do not crash when requested vocab is too high.
        hard_vocab_limit=False,
    )
    return model_prefix + ".model"


def encode_text(sp: spm.SentencePieceProcessor, text: str, max_len: int, bos_id: int, eos_id: int) -> List[int]:
    ids = sp.encode(text, out_type=int)
    ids = ids[: max_len - 2]
    return [bos_id] + ids + [eos_id]


class ParallelTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, src_sp: spm.SentencePieceProcessor, tgt_sp: spm.SentencePieceProcessor, cfg: Config):
        self.src = df["source"].tolist()
        self.tgt = df["target"].tolist()
        self.src_sp = src_sp
        self.tgt_sp = tgt_sp
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int):
        src_ids = encode_text(self.src_sp, self.src[idx], self.cfg.max_len, self.cfg.bos_id, self.cfg.eos_id)
        tgt_ids = encode_text(self.tgt_sp, self.tgt[idx], self.cfg.max_len, self.cfg.bos_id, self.cfg.eos_id)
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
        }


class InferenceTextDataset(Dataset):
    def __init__(self, sources: List[str], src_sp: spm.SentencePieceProcessor, cfg: Config):
        self.sources = sources
        self.src_sp = src_sp
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int):
        src_ids = encode_text(self.src_sp, self.sources[idx], self.cfg.max_len, self.cfg.bos_id, self.cfg.eos_id)
        return src_ids


def collate_fn(batch, pad_id: int):
    src_max = max(len(x["src_ids"]) for x in batch)
    tgt_max = max(len(x["tgt_ids"]) for x in batch)

    src_batch = []
    tgt_batch = []
    for item in batch:
        src = item["src_ids"] + [pad_id] * (src_max - len(item["src_ids"]))
        tgt = item["tgt_ids"] + [pad_id] * (tgt_max - len(item["tgt_ids"]))
        src_batch.append(src)
        tgt_batch.append(tgt)

    src_tensor = torch.tensor(src_batch, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_batch, dtype=torch.long)
    return src_tensor, tgt_tensor


def collate_src_fn(batch, pad_id: int):
    src_max = max(len(x) for x in batch)
    src_batch = [x + [pad_id] * (src_max - len(x)) for x in batch]
    return torch.tensor(src_batch, dtype=torch.long)


# -------------------------
# Model
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.src_emb = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.tgt_emb = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len)
        self.dropout = nn.Dropout(cfg.dropout)

        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            num_encoder_layers=cfg.n_encoder_layers,
            num_decoder_layers=cfg.n_decoder_layers,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.lm_head = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        src = self.dropout(self.pos(self.src_emb(src_ids) * math.sqrt(self.cfg.d_model)))
        tgt = self.dropout(self.pos(self.tgt_emb(tgt_in_ids) * math.sqrt(self.cfg.d_model)))

        out = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.lm_head(out)
        return logits


# -------------------------
# Train / Eval
# -------------------------
def make_tgt_mask(tgt_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))


def make_scheduler(optimizer, warmup_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        return (warmup_steps ** 0.5) / ((step + 1) ** 0.5)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, cfg: Config):
    model.train()
    total_loss = 0.0

    for src_ids, tgt_ids in loader:
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)

        tgt_in = tgt_ids[:, :-1]
        tgt_out = tgt_ids[:, 1:]

        src_pad = src_ids.eq(cfg.pad_id)
        tgt_in_pad = tgt_in.eq(cfg.pad_id)
        tgt_mask = make_tgt_mask(tgt_in.size(1), device)

        logits = model(src_ids, tgt_in, src_pad, tgt_in_pad, tgt_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg: Config):
    model.eval()
    total_loss = 0.0

    for src_ids, tgt_ids in loader:
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)

        tgt_in = tgt_ids[:, :-1]
        tgt_out = tgt_ids[:, 1:]

        src_pad = src_ids.eq(cfg.pad_id)
        tgt_in_pad = tgt_in.eq(cfg.pad_id)
        tgt_mask = make_tgt_mask(tgt_in.size(1), device)

        logits = model(src_ids, tgt_in, src_pad, tgt_in_pad, tgt_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def greedy_decode(model, src_ids, cfg: Config, device: torch.device, max_new_tokens: int = 64):
    model.eval()

    src_ids = src_ids.to(device)
    src_pad = src_ids.eq(cfg.pad_id)

    ys = torch.full((src_ids.size(0), 1), cfg.bos_id, dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        tgt_pad = ys.eq(cfg.pad_id)
        tgt_mask = make_tgt_mask(ys.size(1), device)
        logits = model(src_ids, ys, src_pad, tgt_pad, tgt_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if (next_token == cfg.eos_id).all():
            break

    return ys


# -------------------------
# Main
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Akkadian->English Transformer from scratch")
    parser.add_argument("--train_csv", type=str, default="C:/Users/Spike/OneDrive - National University of Singapore/Desktop/NUS/Improving my Coding/Kaggle Competitions/Akkadian to English (Data)/data/train.csv")
    parser.add_argument("--sentences_csv", type=str, default="")
    parser.add_argument("--test_csv", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--predictions_csv", type=str, default="test_predictions.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--src_vocab_size", type=int, default=12000)
    parser.add_argument("--tgt_vocab_size", type=int, default=12000)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        src_vocab_size=args.src_vocab_size,
        tgt_vocab_size=args.tgt_vocab_size,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing data...")
    full_df = load_parallel_data(args.train_csv, args.sentences_csv)
    train_df, val_df = split_train_val(full_df, cfg.val_ratio, cfg.seed)
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    print("Training SentencePiece tokenizers...")
    src_prefix = str(output_dir / "spm_src")
    tgt_prefix = str(output_dir / "spm_tgt")
    src_model_path = train_sentencepiece(train_df["source"].tolist(), src_prefix, cfg.src_vocab_size, cfg)
    tgt_model_path = train_sentencepiece(train_df["target"].tolist(), tgt_prefix, cfg.tgt_vocab_size, cfg)

    src_sp = spm.SentencePieceProcessor(model_file=src_model_path)
    tgt_sp = spm.SentencePieceProcessor(model_file=tgt_model_path)

    train_ds = ParallelTextDataset(train_df, src_sp, tgt_sp, cfg)
    val_ds = ParallelTextDataset(val_df, src_sp, tgt_sp, cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, cfg.pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, cfg.pad_id),
    )

    model = Seq2SeqTransformer(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(optimizer, cfg.warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_id)

    best_val = float("inf")

    print("Starting training...")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, cfg)
        val_loss = evaluate(model, val_loader, criterion, device, cfg)
        print(f"Epoch {epoch}/{cfg.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = output_dir / "best_model.pt"
            torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, ckpt_path)
            print(f"Saved best model to {ckpt_path}")

    if len(val_df) > 0:
        print("Running a quick inference example from validation set...")
        sample_src = val_df.iloc[0]["source"]
        src_ids = encode_text(src_sp, sample_src, cfg.max_len, cfg.bos_id, cfg.eos_id)
        src_tensor = torch.tensor([src_ids], dtype=torch.long)

        pred_ids = greedy_decode(model, src_tensor, cfg, device, max_new_tokens=64)[0].tolist()
        pred_ids = [tok for tok in pred_ids if tok not in {cfg.pad_id, cfg.bos_id, cfg.eos_id}]
        pred_text = tgt_sp.decode(pred_ids)

        print("Source:", sample_src)
        print("Prediction:", pred_text)
        print("Reference:", val_df.iloc[0]["target"])

    print("Loading test data and generating predictions...")
    test_df, _ = load_test_data(args.train_csv, args.test_csv)
    test_ds = InferenceTextDataset(test_df["source"].tolist(), src_sp, cfg)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_src_fn(b, cfg.pad_id),
    )

    predictions = []
    for src_batch in test_loader:
        decoded = greedy_decode(model, src_batch, cfg, device, max_new_tokens=64).tolist()
        for pred_ids in decoded:
            pred_ids = [tok for tok in pred_ids if tok not in {cfg.pad_id, cfg.bos_id, cfg.eos_id}]
            predictions.append(tgt_sp.decode(pred_ids))

    id_col = _pick_first_existing(list(test_df.columns), KEY_COLUMN_CANDIDATES)
    pred_df = pd.DataFrame({"prediction": predictions})
    if id_col:
        pred_df.insert(0, id_col, test_df[id_col].values)

    pred_path = Path(args.predictions_csv)
    if not pred_path.is_absolute():
        pred_path = output_dir / pred_path
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved test predictions to {pred_path}")


if __name__ == "__main__":
    main()
