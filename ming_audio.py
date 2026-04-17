"""
ming_audio.py 将MingAudio类从cookbooks/test.py中独立出来。
"""
import copy
import contextlib
import json
import os
import re
import warnings
import random
from typing import Any, Generator, Optional, Union

import numpy as np
import torch
import torchaudio
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from prompt_audio_cache import resolve_prompt_audio_path
from sentence_manager.sentence_manager import SentenceNormalizer
from spkemb_extractor import SpkembExtractor


warnings.filterwarnings("ignore")


def _resolve_model_source(model_path: str) -> str:
    if os.path.isdir(model_path):
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshot_dirs = [
                os.path.join(snapshots_dir, name)
                for name in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, name))
            ]
            if snapshot_dirs:
                return max(snapshot_dirs, key=os.path.getmtime)
        return model_path

    local_hf_cache_repo = os.path.join("Models", "hub", model_path)
    if os.path.isdir(local_hf_cache_repo):
        return _resolve_model_source(local_hf_cache_repo)

    return model_path


BASE_CAPTION_TEMPLATE = {
    "audio_sequence": [
        {
            "序号": 1,
            "说话人": "speaker_1",
            "方言": None,
            "风格": None,
            "语速": None,
            "基频": None,
            "音量": None,
            "情感": None,
            "BGM": {
                "Genre": None,
                "Mood": None,
                "Instrument": None,
                "Theme": None,
                "ENV": None,
                "SNR": None,
            },
            "IP": None,
        }
    ]
}


def seed_everything(seed=1895, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


class MingAudio:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = device
        resolved_model_source = _resolve_model_source(model_path)
        self.model = BailingMMNativeForConditionalGeneration.from_pretrained(
            resolved_model_source,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model: Any = self.model.eval()
        model = model.to(torch.bfloat16)
        model = model.to(self.device)
        self.model = model

        if self.model.model_type == "dense":
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_source)
        else:
            repo_root = os.path.dirname(os.path.abspath(__file__))
            self.tokenizer = AutoTokenizer.from_pretrained(repo_root, trust_remote_code=True)
        self.model.tokenizer = self.tokenizer
        self.sample_rate = self.model.config.audio_tokenizer_config.sample_rate
        self.patch_size = self.model.config.ditar_config["patch_size"]
        self.normalizer = self.init_tn_normalizer(tokenizer=self.tokenizer)

        local_model_path = resolved_model_source if os.path.isdir(resolved_model_source) else snapshot_download(repo_id=model_path)
        self.spkemb_extractor = SpkembExtractor(f"{local_model_path}/campplus.onnx")

    def init_tn_normalizer(self, config_file_path: Optional[str] = None, tokenizer=None):
        if config_file_path is None:
            repo_root = os.path.dirname(os.path.abspath(__file__))
            config_file_path = os.path.join(repo_root, "sentence_manager", "default_config.yaml")

        with open(config_file_path, "r", encoding="utf-8") as file:
            self.sentence_manager_config = yaml.safe_load(file)

        if "split_token" not in self.sentence_manager_config:
            self.sentence_manager_config["split_token"] = []

        assert isinstance(self.sentence_manager_config["split_token"], list)
        if tokenizer is not None:
            self.sentence_manager_config["split_token"].append(re.escape(tokenizer.eos_token))

        normalizer = SentenceNormalizer(self.sentence_manager_config.get("text_norm", {}))
        return normalizer

    def create_instruction(self, user_input: dict):
        new_caption = copy.deepcopy(BASE_CAPTION_TEMPLATE)
        target_item_dict = new_caption["audio_sequence"][0]

        for key, value in user_input.items():
            if key in target_item_dict:
                target_item_dict[key] = value

        if target_item_dict["BGM"].get("SNR", None) is not None:
            new_order = ["序号", "说话人", "BGM", "情感", "方言", "风格", "语速", "基频", "音量", "IP"]
            target_item_dict = {k: target_item_dict[k] for k in new_order if k in target_item_dict}
            new_caption["audio_sequence"][0] = target_item_dict

        return new_caption

    def pad_waveform(self, waveform: torch.Tensor):
        pad_align = int(1 / 12.5 * self.patch_size * self.sample_rate)
        new_len = (waveform.size(-1) + pad_align - 1) // pad_align * pad_align
        if new_len != waveform.size(1):
            new_wav = torch.zeros(1, new_len, dtype=waveform.dtype, device=waveform.device)
            new_wav[:, : waveform.size(1)] = waveform.clone()
            waveform = new_wav
        return waveform

    def preprocess_one_prompt_wav(
        self,
        waveform_path: Optional[str],
        use_spk_emb: bool,
        *,
        prompt_audio_diagnostics: list[dict[str, object]] | None = None,
    ):
        if waveform_path is None:
            return None, None

        prompt_audio_diag: dict[str, object] = {}
        resolved_waveform_path = (
            resolve_prompt_audio_path(waveform_path, diagnostics=prompt_audio_diag) if isinstance(waveform_path, str) else waveform_path
        )
        waveform, sr = torchaudio.load(resolved_waveform_path)
        waveform1 = waveform.clone()
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        if use_spk_emb:
            waveform1 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform1)
            spk_emb = self.spkemb_extractor(waveform1)
        else:
            spk_emb = None
        if prompt_audio_diagnostics is not None and prompt_audio_diag:
            prompt_audio_diag["sourcePath"] = waveform_path
            prompt_audio_diag["resolvedPath"] = resolved_waveform_path
            prompt_audio_diagnostics.append(prompt_audio_diag)
        return waveform, spk_emb

    def _prepare_speech_inputs(
        self,
        use_spk_emb: bool,
        use_zero_spk_emb: bool,
        instruction: Optional[dict],
        prompt_wav_path: Optional[Union[str, list]],
        prompt_text: Optional[str],
    ):
        prompt_audio_diagnostics: list[dict[str, object]] = []
        if prompt_wav_path is None:
            prompt_waveform, prompt_text, spk_emb = None, None, None
            if use_zero_spk_emb:
                spk_emb = [torch.zeros(1, 192, device=self.device, dtype=torch.bfloat16)]
        else:
            paths = prompt_wav_path if isinstance(prompt_wav_path, list) else [prompt_wav_path]
            processed_prompts = [
                self.preprocess_one_prompt_wav(path, use_spk_emb, prompt_audio_diagnostics=prompt_audio_diagnostics)
                for path in paths
            ]
            waveforms_list, spk_emb = zip(*processed_prompts)
            prompt_waveform = torch.cat(waveforms_list, dim=-1)
            prompt_waveform = self.pad_waveform(prompt_waveform)
            spk_emb = list(spk_emb)
            if all(item is None for item in spk_emb):
                spk_emb = None

        instruction_payload: Optional[str] = None
        if instruction is not None:
            instruction_obj = self.create_instruction(instruction)
            instruction_payload = json.dumps(instruction_obj, ensure_ascii=False)

        prompt_audio_summary = {
            "promptAudioCount": len(prompt_audio_diagnostics),
            "remotePromptAudioCount": len([item for item in prompt_audio_diagnostics if item.get("remote") is True]),
            "promptAudioCacheHitCount": len([item for item in prompt_audio_diagnostics if item.get("cacheHit") is True]),
            "promptAudioDownloadMs": round(
                sum(float(item.get("downloadMs") or 0.0) for item in prompt_audio_diagnostics),
                2,
            ),
            "promptAudioLockWaitMs": round(
                sum(float(item.get("lockWaitMs") or 0.0) for item in prompt_audio_diagnostics),
                2,
            ),
            "promptAudioDiagnostics": prompt_audio_diagnostics,
        }

        return prompt_waveform, prompt_text, spk_emb, instruction_payload, prompt_audio_summary

    @torch.inference_mode()
    def speech_generation(
        self,
        prompt: str,
        text: str,
        use_spk_emb: bool = False,
        use_zero_spk_emb: bool = False,
        instruction: Optional[dict] = None,
        prompt_wav_path: Optional[Union[str, list]] = None,
        prompt_text: Optional[str] = None,
        max_decode_steps: int = 200,
        cfg: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0.0,
        output_wav_path: Optional[str] = None,
        diagnostics: Optional[dict[str, Any]] = None,
    ):
        prompt_waveform, prompt_text, spk_emb, instruction_payload, prompt_audio_summary = self._prepare_speech_inputs(
            use_spk_emb=use_spk_emb,
            use_zero_spk_emb=use_zero_spk_emb,
            instruction=instruction,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
        )
        generation_diagnostics: dict[str, Any] = {}

        model: Any = self.model
        waveform = model.generate(
            prompt=prompt,
            text=text,
            spk_emb=spk_emb,
            instruction=instruction_payload,
            prompt_waveform=prompt_waveform,
            prompt_text=prompt_text,
            max_decode_steps=max_decode_steps,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
            use_zero_spk_emb=use_zero_spk_emb,
            diagnostics=generation_diagnostics,
        )

        if output_wav_path is not None:
            output_dir = os.path.dirname(output_wav_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torchaudio.save(output_wav_path, waveform, sample_rate=self.sample_rate)
        if diagnostics is not None:
            diagnostics.update(prompt_audio_summary)
            diagnostics.update(generation_diagnostics)
        return waveform

    @torch.inference_mode()
    def speech_generation_stream(
        self,
        prompt: str,
        text: str,
        use_spk_emb: bool = False,
        use_zero_spk_emb: bool = False,
        instruction: Optional[dict] = None,
        prompt_wav_path: Optional[Union[str, list]] = None,
        prompt_text: Optional[str] = None,
        max_decode_steps: int = 200,
        cfg: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0.0,
        diagnostics: Optional[dict[str, Any]] = None,
    ) -> Generator[bytes, None, None]:
        prompt_waveform, prompt_text, spk_emb, instruction_payload, prompt_audio_summary = self._prepare_speech_inputs(
            use_spk_emb=use_spk_emb,
            use_zero_spk_emb=use_zero_spk_emb,
            instruction=instruction,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
        )
        generation_diagnostics: dict[str, Any] = {}

        model: Any = self.model
        stream_state = (None, None, None)
        past_key_values = None
        use_cache = True

        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if torch.cuda.is_available() else contextlib.nullcontext()
        with autocast_ctx:
            for sampled_tokens, last_chunk in model.sample(
                prompt=prompt,
                text=text,
                spk_emb=spk_emb,
                instruction=instruction_payload,
                prompt_waveform=prompt_waveform,
                prompt_text=prompt_text,
                max_decode_steps=max_decode_steps,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
                use_zero_spk_emb=use_zero_spk_emb,
                diagnostics=generation_diagnostics,
            ):
                speech_chunk, stream_state, past_key_values = model.audio.decode(
                    sampled_tokens,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    stream_state=stream_state,
                    last_chunk=last_chunk,
                )

                pcm_tensor = torch.clamp(speech_chunk.squeeze(0), -1.0, 1.0)
                pcm_tensor = pcm_tensor.mul_(32767.0).to(torch.int16).contiguous().cpu().view(-1)
                if pcm_tensor.numel() == 0:
                    continue
                chunk_pcm = pcm_tensor.numpy().tobytes()
                if chunk_pcm:
                    if diagnostics is not None and not diagnostics:
                        diagnostics.update(prompt_audio_summary)
                        diagnostics.update(generation_diagnostics)
                    yield chunk_pcm

    def generation(
        self,
        prompt: str,
        text: str,
        max_decode_steps: int = 200,
    ):
        model: Any = self.model
        text = model.generate_text(
            prompt=prompt,
            text=text,
            max_decode_steps=max_decode_steps,
        )
        return text
