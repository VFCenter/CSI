# Adjudicator Agent Text Features

## Purpose

`utils/agent__feature_extractor.py` extracts features only from text generated
by the upstream agents. It does not extract original news-text features,
sentiment features, raw visual-frame features, or raw audio features.

The extractor runs after the Multimodal Forensics Unit and the Case Review
Unit have completed. Each video produces one PyTorch pickle file:

```text
features/<dataset>/<video_id>.pkl
```

The directory can be changed with the `CSI_FEATURE_DIR` environment variable
or the CLI `--output-dir` argument. Files are written with `torch.save` and
loaded with `torch.load`.

## Feature Definitions

| Agent output | Accepted JSON fields | Saved tensor key | Adjudicator mapping | Tensor shape |
| --- | --- | --- | --- | --- |
| Intelligence Analyst text analysis | `E`, `text_analysis`, `intelligence_analysis`, `official_reports` | `text_analysis_fea` | Loaded as `structed_title_fea` and concatenated into `all_combine_text_fea` | FakeTT: `[512, 512]`; FakeSV: `[512, 768]` |
| Acoustic Analyst text analysis | `P_audio`, `audio_analysis`, `acoustic_analysis` | `audio_analysis_fea` | `all_audio_analysis_fea` | FakeTT: `[512, 512]`; FakeSV: `[512, 768]` |
| Visual Analyst text analysis | `P_vision`, `visual_analysis`, `vision_analysis` | `visual_analysis_fea` | `all_visual_analysis_fea` | FakeTT: `[512, 512]`; FakeSV: `[512, 768]` |
| Review Team final deliberation | `review_result`, `final_decision`, `reasoning_results`, `analysis`, `meeting_log` | `review_result_fea` | `all_review_result_fea` | FakeTT: `[512, 512]`; FakeSV: `[512, 768]` |

`model.py` does not define a separately named `text_analysis` argument. The
dataloader therefore maps `text_analysis_fea` to its existing
`structed_title_fea` slot. During collation this feature is combined with the
pre-existing raw news-text feature to form `all_combine_text_fea`.

## Extraction Models

| Dataset | Language | Hugging Face model | Hidden dimension |
| --- | --- | --- | --- |
| FakeTT | English | `openai/clip-vit-base-patch32` | 512 |
| FakeSV | Chinese | `OFA-Sys/chinese-clip-vit-base-patch16` | 768 |

The extractor uses the CLIP text encoder's `last_hidden_state`, not the final
projected CLIP embedding. This is required because the FakeSV adjudicator
expects the Chinese-CLIP text hidden size of 768, while the projected CLIP
embedding is 512-dimensional.

Each model first tokenizes its four text fields using the tokenizer's supported
context length. The resulting hidden-state sequence is converted to
`float32`, moved to CPU, and padded with zeros or truncated to 512 tokens.

## Saved Pickle Structure

Each `.pkl` file contains:

```python
{
    "video_id": str,
    "dataset": "fakett" | "fakesv",
    "clip_model": str,
    "text_dim": 512 | 768,
    "sequence_length": 512,
    "text_analysis_fea": torch.Tensor,
    "audio_analysis_fea": torch.Tensor,
    "visual_analysis_fea": torch.Tensor,
    "review_result_fea": torch.Tensor,
    "text_fields": {
        "text_analysis": str,
        "audio_analysis": str,
        "visual_analysis": str,
        "review_result": str,
    },
}
```

`load_saved_agent_features()` validates that all four feature tensors exist and
that every tensor has shape `[512, text_dim]` before returning the dictionary.

## Automatic Pipeline Integration

`csi_framework.execute_csi_multi_agent()` invokes the extractor after the MFU
and Review Team finish. It passes these fields:

```python
{
    "text_analysis": acq_results["E"],
    "audio_analysis": acq_results["P_audio"],
    "visual_analysis": acq_results["P_vision"],
    "review_result": delib_results["final_decision"],
}
```

On success, the returned CSI result contains:

```python
"adjudicator_features": {
    "status": "ok",
    "path": "features/<dataset>/<video_id>.pkl",
}
```

If model loading or extraction fails, `status` is `error` and the error message
is retained without discarding the upstream agent result.

## Extract Existing JSON Results

```bash
python agents/case_review_unit/adjudicator/utils/agent__feature_extractor.py \
  --dataset fakett \
  --input results_FakeTT_csi.json \
  --output-dir features
```

Supported input layouts are a JSON object keyed by video ID, a JSON list, a
single JSON object containing `video_id`, and JSON Lines.

Use cached Hugging Face models only:

```bash
python agents/case_review_unit/adjudicator/utils/agent__feature_extractor.py \
  --dataset fakesv \
  --input results_FakeSV_csi.json \
  --output-dir features \
  --local-files-only
```

## Python API

```python
from agents.case_review_unit.adjudicator.utils.agent__feature_extractor import (
    extract_and_save_agent_features,
)

path = extract_and_save_agent_features(
    video_id="example-video",
    dataset="FakeTT",
    record={
        "text_analysis": "Intelligence Analyst output",
        "audio_analysis": "Acoustic Analyst output",
        "visual_analysis": "Visual Analyst output",
        "review_result": "Review Team final deliberation",
    },
    output_dir="features",
)
```

## Dependencies

The extractor requires PyTorch and Transformers. Install the project
dependencies before running it:

```bash
pip install -r requirements.txt
```
