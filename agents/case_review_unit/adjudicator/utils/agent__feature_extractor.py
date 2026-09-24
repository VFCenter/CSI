from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "fakett": {
        "clip_model": "openai/clip-vit-base-patch32",
        "text_dim": 512,
    },
    "fakesv": {
        "clip_model": "OFA-Sys/chinese-clip-vit-base-patch16",
        "text_dim": 768,
    },
}

SEQUENCE_LENGTH = 512
FEATURE_KEYS = (
    "text_analysis_fea",
    "audio_analysis_fea",
    "visual_analysis_fea",
    "review_result_fea",
)


def normalize_dataset(dataset: str) -> str:
    key = str(dataset).strip().lower()
    if key not in DATASET_CONFIG:
        raise ValueError(
            f"Unsupported dataset {dataset!r}; expected one of "
            f"{', '.join(sorted(DATASET_CONFIG))}."
        )
    return key


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, Mapping):
        parts = []
        for key, item in value.items():
            item_text = _to_text(item)
            if item_text:
                parts.append(f"{key}: {item_text}")
        return "\n".join(parts)
    if isinstance(value, (list, tuple)):
        return "\n".join(text for item in value if (text := _to_text(item)))
    return str(value).strip()


def _first_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        text = _to_text(record.get(key))
        if text:
            return text
    for nested_key in ("analysis_result", "agent_response", "results", "result"):
        nested = record.get(nested_key)
        if isinstance(nested, Mapping):
            text = _first_text(nested, keys)
            if text:
                return text
    return ""


def extract_agent_text_fields(record: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "text_analysis": _first_text(
            record,
            ("E", "text_analysis", "intelligence_analysis", "official_reports"),
        ),
        "audio_analysis": _first_text(
            record,
            ("P_audio", "audio_analysis", "acoustic_analysis"),
        ),
        "visual_analysis": _first_text(
            record,
            ("P_vision", "visual_analysis", "vision_analysis"),
        ),
        "review_result": _first_text(
            record,
            (
                "review_result",
                "final_decision",
                "reasoning_results",
                "analysis",
                "meeting_log",
            ),
        ),
    }


def _load_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Agent feature extraction requires transformers. Install it with "
            '`pip install transformers>=4.36`.'
        ) from exc
    return AutoModel, AutoTokenizer


class AgentFeatureExtractor:
    def __init__(
        self,
        dataset: str,
        device: Optional[str | torch.device] = None,
        local_files_only: bool = False,
    ) -> None:
        self.dataset = normalize_dataset(dataset)
        self.config = DATASET_CONFIG[self.dataset]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        AutoModel, AutoTokenizer = _load_transformers()
        model_name = self.config["clip_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model.to(self.device).eval()
        text_config = getattr(self.model.config, "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size != self.config["text_dim"]:
            raise ValueError(
                f"{model_name} exposes text hidden size {hidden_size}, expected "
                f"{self.config['text_dim']} for {self.dataset}."
            )

    def _tokenize(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        max_length = getattr(self.tokenizer, "model_max_length", 77)
        if not isinstance(max_length, int) or max_length > 2048:
            max_length = 77
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    @staticmethod
    def _pad_sequence(hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden[:SEQUENCE_LENGTH]
        if hidden.shape[0] < SEQUENCE_LENGTH:
            hidden = torch.cat(
                (
                    hidden,
                    torch.zeros(
                        SEQUENCE_LENGTH - hidden.shape[0],
                        hidden.shape[1],
                        dtype=hidden.dtype,
                        device=hidden.device,
                    ),
                ),
                dim=0,
            )
        return hidden

    def encode_fields(self, fields: Mapping[str, str]) -> Dict[str, torch.Tensor]:
        ordered = [
            fields.get("text_analysis", ""),
            fields.get("audio_analysis", ""),
            fields.get("visual_analysis", ""),
            fields.get("review_result", ""),
        ]
        encoded = self._tokenize(ordered)
        text_model = getattr(self.model, "text_model", self.model)
        with torch.no_grad():
            hidden = text_model(**encoded, return_dict=True).last_hidden_state
        hidden = hidden.detach().cpu().float()
        hidden = torch.stack([self._pad_sequence(item) for item in hidden], dim=0)
        return {
            "text_analysis_fea": hidden[0],
            "audio_analysis_fea": hidden[1],
            "visual_analysis_fea": hidden[2],
            "review_result_fea": hidden[3],
        }

    def encode_record(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        fields = extract_agent_text_fields(record)
        return {
            "features": self.encode_fields(fields),
            "text_fields": fields,
        }

    def encode_and_save(
        self,
        video_id: str,
        record: Mapping[str, Any],
        output_dir: str | os.PathLike[str] = "features",
    ) -> Path:
        result = self.encode_record(record)
        output_path = Path(output_dir) / self.dataset / f"{video_id}.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "video_id": str(video_id),
            "dataset": self.dataset,
            "clip_model": self.config["clip_model"],
            "text_dim": self.config["text_dim"],
            "sequence_length": SEQUENCE_LENGTH,
            **result["features"],
            "text_fields": result["text_fields"],
        }
        torch.save(payload, output_path)
        return output_path


_EXTRACTOR_CACHE: Dict[Tuple[str, str], AgentFeatureExtractor] = {}


def get_agent_feature_extractor(
    dataset: str,
    device: Optional[str | torch.device] = None,
    local_files_only: bool = False,
) -> AgentFeatureExtractor:
    normalized = normalize_dataset(dataset)
    device_key = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_key = (normalized, device_key)
    if cache_key not in _EXTRACTOR_CACHE:
        _EXTRACTOR_CACHE[cache_key] = AgentFeatureExtractor(
            normalized,
            device=device,
            local_files_only=local_files_only,
        )
    return _EXTRACTOR_CACHE[cache_key]


def extract_and_save_agent_features(
    video_id: str,
    dataset: str,
    record: Mapping[str, Any],
    output_dir: str | os.PathLike[str] = "features",
    device: Optional[str | torch.device] = None,
    local_files_only: bool = False,
) -> Path:
    extractor = get_agent_feature_extractor(
        dataset,
        device=device,
        local_files_only=local_files_only,
    )
    return extractor.encode_and_save(str(video_id), record, output_dir=output_dir)


def load_saved_agent_features(
    feature_path: str | os.PathLike[str],
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(feature_path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Feature file must contain a dictionary: {feature_path}")
    missing = [key for key in FEATURE_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Feature file is missing keys {missing}: {feature_path}")
    text_dim = int(payload["text_dim"])
    for key in FEATURE_KEYS:
        tensor = payload[key]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (
            SEQUENCE_LENGTH,
            text_dim,
        ):
            raise ValueError(
                f"{key} must have shape [{SEQUENCE_LENGTH}, {text_dim}], "
                f"got {getattr(tensor, 'shape', None)}."
            )
    return dict(payload)


def _read_json_records(path: str | os.PathLike[str]) -> List[Tuple[str, Mapping[str, Any]]]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(value, Mapping):
        if "video_id" in value:
            value = [value]
        else:
            value = [
                dict(item, video_id=video_id) if isinstance(item, Mapping) else {}
                for video_id, item in value.items()
            ]
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON object, list, or JSON-lines file: {path}")
    records = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        video_id = item.get("video_id") or item.get("vid") or item.get("id")
        if video_id is not None:
            records.append((str(video_id), item))
    return records


def extract_agent_features_from_json(
    input_path: str | os.PathLike[str],
    dataset: str,
    output_dir: str | os.PathLike[str] = "features",
    device: Optional[str | torch.device] = None,
    local_files_only: bool = False,
) -> List[Path]:
    extractor = get_agent_feature_extractor(
        dataset,
        device=device,
        local_files_only=local_files_only,
    )
    return [
        extractor.encode_and_save(video_id, record, output_dir=output_dir)
        for video_id, record in _read_json_records(input_path)
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract CSI agent output text features")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="features")
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = extract_agent_features_from_json(
        args.input,
        args.dataset,
        output_dir=args.output_dir,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    print(f"Saved {len(paths)} feature file(s) under {Path(args.output_dir) / args.dataset.lower()}")


if __name__ == "__main__":
    main()
