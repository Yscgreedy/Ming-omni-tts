---
license: apache-2.0
pipeline_tag: text-to-speech
---

<p align="center">🌐<a href="https://xqacmer.github.io/Ming-omni-tts/">Project Page</a> ｜🤗 <a href="https://huggingface.co/inclusionAI/Ming-omni-tts-0.5B">Hugging Face</a>｜ 🤖 <a href="https://modelscope.cn/models/inclusionAI/Ming-omni-tts-0.5B">ModelScope</a> | 🎮 <a href="https://modelscope.cn/studios/antsipan/ming-uniaudio-demo">Gradio Demo-zh</a> | 🎮 <a href="https://huggingface.co/spaces/cafe3310/ming-uniaudio-demo-en">Gradio Demo-en</a> | 💬 <a href="https://qr.dingtalk.com/action/joingroup?code=v1,k1,O7MiZqrOrB2c7PnUZVBNvmDh/6tghNLMEtXqMteyRIpuRVJIwrSsXmL8oFqU5ajJ&_dt_no_comment=1&origin=11? ">DingTalk(钉钉)</a>

## Introduction

Ming-omni-tts is a high-performance unified audio generation model that achieves precise control over speech attributes and enables single-channel synthesis of speech, environmental sounds, and music. Powered by a custom 12.5Hz continuous tokenizer and Patch-by-Patch compression, it delivers competitive inference efficiency (3.1Hz). Additionally, the model features robust text normalization capabilities for the accurate and natural narration of complex mathematical and chemical expressions.

<strong>🚀 Core Capabilities</strong>

- 🔊 **Fine-grained Vocal Control:** The model supports precise control over speech rate, pitch, volume, emotion, and dialect through simple commands. Notably, its accuracy for Cantonese dialect control is as high as **93%**, and its emotion control accuracy reaches **46.7%**, surpassing CosyVoice3.
- 🌌  **Intelligent Voice Design:** Features 100+ premium built-in voices and supports zero-shot voice design through natural language descriptions. Its performance on the Instruct-TTS-Eval-zh benchmark is on par with Qwen3-TTS.
- 🎶  **Immersive Unified Generation:** The industry’s first autoregressive model to jointly generate speech, ambient sound, and music in a single channel. Built on a custom 12.5Hz continuous tokenizer and a DiT head architecture, it delivers a seamless, "in-the-scene" auditory experience.
- ⚡ **High-efficiency Inference:** Introduces a "Patch-by-Patch" compression strategy that reduces the LLM inference frame rate to 3.1Hz. This significantly cuts latency and enables podcast-style audio generation while preserving naturalness and audio detail.
- 🧪 **Professional Text Normalization:** The model accurately parses and narrates complex formats, including mathematical expressions and chemical equations, ensuring natural-sounding output for specialized applications.

##  Evaluation
- **Reconstruction:** The 12Hz tokenizer supports high-quality reconstruction across speech, music, and sound. Its performance is comparable to existing state-of-the-art methods across key fidelity metrics.
- **Dialect Generation:** Achieves **96%** accuracy on WSYue-TTS-Eval and **86%** WSC-TTS-Eval, outperforming CosyVoice3.
- **Emotional Expressiveness:** Delivers an average accuracy of **76.7%** on CV3-Eval emotional sets and **46.7%** on neutral emotion sets, significantly surpassing CosyVoice3-Base (40%) to reach SOTA levels.
- **Instruction-based Voice Design:** Scores **76.20%** on InstructTTS-Eval-ZH. Its instruction-following capability is on par with Qwen3-TTS-VoiceDesign.
- **Zero-shot Voice Clone:** Exhibits exceptional stability on Seed-tts-eval (Chinese) with a WER of **0.83%**, outperforming SeedTTS and GLM-TTS.
- **Text Normalization (TN):** On internal technical testsets, the model achieves a CER of **1.97%** in normalized regions, delivering performance comparable to Gemini-2.5 Pro.

## Example Usage

```bash
git clone https://github.com/inclusionAI/Ming-omni-tts.git
cd Ming-omni-tts
python3 cookbooks/test.py
```