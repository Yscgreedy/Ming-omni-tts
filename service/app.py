import asyncio
import io
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union
import sys
import soundfile as sf
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ming_audio import MingAudio, seed_everything


class GenerateRequest(BaseModel):
    task_type: Literal["speech", "text"] = Field(default="speech", description="speech: 音频生成, text: 文本生成")
    prompt: str = Field(..., description="系统提示词")
    text: str = Field(..., description="输入文本")

    use_spk_emb: bool = False
    use_zero_spk_emb: bool = False
    instruction: Optional[Dict[str, Any]] = None
    prompt_wav_path: Optional[Union[str, List[str]]] = None
    prompt_text: Optional[str] = None
    max_decode_steps: int = 200
    cfg: float = 2.0
    sigma: float = 0.25
    temperature: float = 0.0

    response_mode: Literal["stream", "path"] = Field(default="stream", description="仅 speech 生效")
    output_wav_path: Optional[str] = Field(default=None, description="response_mode=path 时可指定")


class StreamRequest(BaseModel):
    prompt: str = Field(..., description="系统提示词")
    text: str = Field(..., description="输入文本")

    use_spk_emb: bool = False
    use_zero_spk_emb: bool = False
    instruction: Optional[Dict[str, Any]] = None
    prompt_wav_path: Optional[Union[str, List[str]]] = None
    prompt_text: Optional[str] = None
    max_decode_steps: int = 200
    cfg: float = 2.0
    sigma: float = 0.25
    temperature: float = 0.0


app = FastAPI(title="Ming Omni TTS Service", version="1.0.0")
model_lock = asyncio.Lock()
model_instance: Optional[MingAudio] = None


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


def _configure_inference_performance() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


@app.post("/seed")
async def seed(request: Dict[str, int]):
    deterministic = bool(request.get("deterministic", False))
    seed_value = int(request["seed"])
    seed_everything(seed_value, deterministic=deterministic)
    return {"status": "seed set", "seed": seed_value, "deterministic": deterministic}

def get_model() -> MingAudio:
    global model_instance
    if model_instance is None:
        default_local_model_path = os.path.join("Models", "hub", "models--inclusionAI--Ming-omni-tts-0.5B")
        default_model_path = default_local_model_path if os.path.isdir(default_local_model_path) else "inclusionAI/Ming-omni-tts-0.5B"
        model_path = os.getenv("MODEL_PATH", default_model_path)
        device = os.getenv("DEVICE", "cuda:0")
        _configure_inference_performance()
        seed_everything(1895, deterministic=False)
        model_instance = MingAudio(model_path=model_path, device=device)
    return model_instance


def _waveform_to_wav_bytes(waveform, sample_rate: int) -> bytes:
    audio = waveform.squeeze(0).detach().cpu().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


async def _iter_pcm_chunks(generator) -> AsyncIterator[bytes]:
    loop = asyncio.get_running_loop()
    _sentinel = object()
    while True:
        chunk = await loop.run_in_executor(None, next, generator, _sentinel)
        if chunk is _sentinel:
            break
        yield chunk


@app.post("/v1/generate")
async def generate(request: GenerateRequest):
    model = get_model()

    try:
        async with model_lock:
            if request.task_type == "text":
                content = model.generation(
                    prompt=request.prompt,
                    text=request.text,
                    max_decode_steps=request.max_decode_steps,
                )
                return JSONResponse({"task_type": "text", "text": content})

            if request.response_mode == "path":
                output_wav_path = request.output_wav_path
                if not output_wav_path:
                    output_wav_path = os.path.join("output", "service", f"{uuid.uuid4().hex}.wav")

                waveform = model.speech_generation(
                    prompt=request.prompt,
                    text=request.text,
                    use_spk_emb=request.use_spk_emb,
                    use_zero_spk_emb=request.use_zero_spk_emb,
                    instruction=request.instruction,
                    prompt_wav_path=request.prompt_wav_path,
                    prompt_text=request.prompt_text,
                    max_decode_steps=request.max_decode_steps,
                    cfg=request.cfg,
                    sigma=request.sigma,
                    temperature=request.temperature,
                    output_wav_path=output_wav_path,
                )
                duration = float(waveform.shape[-1]) / float(model.sample_rate)
                return JSONResponse(
                    {
                        "task_type": "speech",
                        "response_mode": "path",
                        "audio_path": output_wav_path,
                        "sample_rate": model.sample_rate,
                        "duration_seconds": duration,
                    }
                )

            waveform = model.speech_generation(
                prompt=request.prompt,
                text=request.text,
                use_spk_emb=request.use_spk_emb,
                use_zero_spk_emb=request.use_zero_spk_emb,
                instruction=request.instruction,
                prompt_wav_path=request.prompt_wav_path,
                prompt_text=request.prompt_text,
                max_decode_steps=request.max_decode_steps,
                cfg=request.cfg,
                sigma=request.sigma,
                temperature=request.temperature,
                output_wav_path=None,
            )
            buffer = io.BytesIO(_waveform_to_wav_bytes(waveform, model.sample_rate))
            buffer.seek(0)

            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": "inline; filename=generated.wav"},
            )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/stream")
async def stream_generate(request: StreamRequest):
    model = get_model()

    try:
        async def _streamer() -> AsyncIterator[bytes]:
            async with model_lock:
                generator = model.speech_generation_stream(
                    prompt=request.prompt,
                    text=request.text,
                    use_spk_emb=request.use_spk_emb,
                    use_zero_spk_emb=request.use_zero_spk_emb,
                    instruction=request.instruction,
                    prompt_wav_path=request.prompt_wav_path,
                    prompt_text=request.prompt_text,
                    max_decode_steps=request.max_decode_steps,
                    cfg=request.cfg,
                    sigma=request.sigma,
                    temperature=request.temperature,
                )
                async for chunk in _iter_pcm_chunks(generator):
                    yield chunk

        return StreamingResponse(
            _streamer(),
            media_type="application/octet-stream",
            headers={
                "X-Audio-Codec": "pcm_s16le",
                "X-Audio-Sample-Rate": str(model.sample_rate),
                "X-Audio-Channels": "1",
                "Content-Disposition": "inline; filename=stream_generated.pcm",
            },
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("service.app:app", host=host, port=port, reload=False)
