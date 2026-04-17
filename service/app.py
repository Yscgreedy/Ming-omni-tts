import asyncio
import io
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union
import sys
import soundfile as sf
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ming_audio import MingAudio, seed_everything


logger = logging.getLogger(__name__)


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
SERVICE_PORT = os.getenv("PORT", "8000")


def _round_ms(value: float | None) -> str:
    if value is None:
        return "0"
    return str(round(float(value), 2))


def _build_trace_headers(*, trace_id: str, diagnostics: dict[str, Any], audio_bytes: int = 0) -> dict[str, str]:
    return {
        "X-Request-Id": trace_id,
        "X-Ming-Pid": str(os.getpid()),
        "X-Ming-Port": SERVICE_PORT,
        "X-Lock-Wait-Ms": _round_ms(diagnostics.get("lockWaitMs")),
        "X-Lock-Held-Ms": _round_ms(diagnostics.get("lockHeldMs")),
        "X-Prompt-Audio-Download-Ms": _round_ms(diagnostics.get("promptAudioDownloadMs")),
        "X-Response-Write-Ms": _round_ms(diagnostics.get("responseWriteMs")),
        "X-Model-Generate-Ms": _round_ms(diagnostics.get("modelGenerateMs")),
        "X-Audio-Bytes": str(audio_bytes),
        "X-Text-Len": str(diagnostics.get("textLen", 0)),
        "X-Max-Decode-Steps": str(diagnostics.get("maxDecodeSteps", 0)),
        "X-Decode-Steps": str(diagnostics.get("decodeSteps", 0)),
        "X-Step-Limit-Reached": str(bool(diagnostics.get("stepLimitReached", False))).lower(),
    }


def _log_request(stage: str, *, trace_id: str, diagnostics: dict[str, Any], **fields: object) -> None:
    logger.info(
        "Ming service trace %s",
        {
            "stage": stage,
            "traceId": trace_id,
            "pid": os.getpid(),
            "port": SERVICE_PORT,
            "textLen": diagnostics.get("textLen"),
            "maxDecodeSteps": diagnostics.get("maxDecodeSteps"),
            "lockWaitMs": diagnostics.get("lockWaitMs"),
            "lockHeldMs": diagnostics.get("lockHeldMs"),
            "promptAudioDownloadMs": diagnostics.get("promptAudioDownloadMs"),
            "promptAudioLockWaitMs": diagnostics.get("promptAudioLockWaitMs"),
            "modelGenerateMs": diagnostics.get("modelGenerateMs"),
            "responseWriteMs": diagnostics.get("responseWriteMs"),
            "audioBytes": diagnostics.get("audioBytes"),
            "decodeSteps": diagnostics.get("decodeSteps"),
            "stepLimitReached": diagnostics.get("stepLimitReached"),
            **fields,
        },
    )


@app.on_event("startup")
async def preload_model_on_startup() -> None:
    async with model_lock:
        get_model()


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
async def generate(http_request: Request, request: GenerateRequest):
    model = get_model()
    trace_id = http_request.headers.get("X-HearOpus-Trace-Id", "").strip() or f"trace_{uuid.uuid4().hex[:12]}"
    diagnostics: dict[str, Any] = {
        "textLen": len(request.text.strip()),
        "maxDecodeSteps": request.max_decode_steps,
    }
    request_started_at = time.perf_counter()
    lock_wait_started_at = request_started_at

    try:
        async with model_lock:
            diagnostics["lockWaitMs"] = round((time.perf_counter() - lock_wait_started_at) * 1000, 2)
            lock_held_started_at = time.perf_counter()
            _log_request("lock_acquired", trace_id=trace_id, diagnostics=diagnostics, taskType=request.task_type, responseMode=request.response_mode)
            if request.task_type == "text":
                content = model.generation(
                    prompt=request.prompt,
                    text=request.text,
                    max_decode_steps=request.max_decode_steps,
                )
                diagnostics["lockHeldMs"] = round((time.perf_counter() - lock_held_started_at) * 1000, 2)
                _log_request("text_complete", trace_id=trace_id, diagnostics=diagnostics)
                return JSONResponse({"task_type": "text", "text": content})

            if request.response_mode == "path":
                output_wav_path = request.output_wav_path
                if not output_wav_path:
                    output_wav_path = os.path.join("output", "service", f"{uuid.uuid4().hex}.wav")

                model_started_at = time.perf_counter()
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
                    diagnostics=diagnostics,
                )
                diagnostics["modelGenerateMs"] = round((time.perf_counter() - model_started_at) * 1000, 2)
                diagnostics["lockHeldMs"] = round((time.perf_counter() - lock_held_started_at) * 1000, 2)
                duration = float(waveform.shape[-1]) / float(model.sample_rate)
                _log_request("path_complete", trace_id=trace_id, diagnostics=diagnostics, outputPath=output_wav_path, durationSeconds=duration)
                return JSONResponse(
                    {
                        "task_type": "speech",
                        "response_mode": "path",
                        "audio_path": output_wav_path,
                        "sample_rate": model.sample_rate,
                        "duration_seconds": duration,
                    }
                )

            model_started_at = time.perf_counter()
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
                diagnostics=diagnostics,
            )
            diagnostics["modelGenerateMs"] = round((time.perf_counter() - model_started_at) * 1000, 2)
            write_started_at = time.perf_counter()
            wav_bytes = _waveform_to_wav_bytes(waveform, model.sample_rate)
            diagnostics["responseWriteMs"] = round((time.perf_counter() - write_started_at) * 1000, 2)
            diagnostics["audioBytes"] = len(wav_bytes)
            diagnostics["lockHeldMs"] = round((time.perf_counter() - lock_held_started_at) * 1000, 2)
            buffer = io.BytesIO(wav_bytes)
            buffer.seek(0)
            _log_request("generate_complete", trace_id=trace_id, diagnostics=diagnostics, totalRequestMs=round((time.perf_counter() - request_started_at) * 1000, 2))

            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "inline; filename=generated.wav",
                    **_build_trace_headers(trace_id=trace_id, diagnostics=diagnostics, audio_bytes=len(wav_bytes)),
                },
            )

    except Exception as exc:
        diagnostics["lockHeldMs"] = round((time.perf_counter() - request_started_at) * 1000, 2)
        _log_request("generate_failed", trace_id=trace_id, diagnostics=diagnostics, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/stream")
async def stream_generate(http_request: Request, request: StreamRequest):
    model = get_model()
    trace_id = http_request.headers.get("X-HearOpus-Trace-Id", "").strip() or f"trace_{uuid.uuid4().hex[:12]}"
    diagnostics: dict[str, Any] = {
        "textLen": len(request.text.strip()),
        "maxDecodeSteps": request.max_decode_steps,
    }

    try:
        async def _streamer() -> AsyncIterator[bytes]:
            lock_wait_started_at = time.perf_counter()
            async with model_lock:
                diagnostics["lockWaitMs"] = round((time.perf_counter() - lock_wait_started_at) * 1000, 2)
                lock_held_started_at = time.perf_counter()
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
                    diagnostics=diagnostics,
                )
                chunk_count = 0
                total_bytes = 0
                async for chunk in _iter_pcm_chunks(generator):
                    chunk_count += 1
                    total_bytes += len(chunk)
                    yield chunk
                diagnostics["audioBytes"] = total_bytes
                diagnostics["chunkCount"] = chunk_count
                diagnostics["lockHeldMs"] = round((time.perf_counter() - lock_held_started_at) * 1000, 2)
                _log_request("stream_complete", trace_id=trace_id, diagnostics=diagnostics)

        return StreamingResponse(
            _streamer(),
            media_type="application/octet-stream",
            headers={
                "X-Audio-Codec": "pcm_s16le",
                "X-Audio-Sample-Rate": str(model.sample_rate),
                "X-Audio-Channels": "1",
                "Content-Disposition": "inline; filename=stream_generated.pcm",
                **_build_trace_headers(trace_id=trace_id, diagnostics=diagnostics, audio_bytes=0),
            },
        )

    except Exception as exc:
        _log_request("stream_failed", trace_id=trace_id, diagnostics=diagnostics, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("service.app:app", host=host, port=port, reload=False)
