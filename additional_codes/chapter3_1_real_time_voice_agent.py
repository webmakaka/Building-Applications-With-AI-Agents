# ！pip install fastapi uvicorn aiohttp python-dotenv websockets

# server.py
import os
import json
import base64
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
from aiohttp import ClientWSTimeout

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 必须设置
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in env or .env")

# OpenAI realtime endpoint + model
# OPENAI_WSS = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
OPENAI_WSS = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2025-06-03"


# 代理设置：可用 http://127.0.0.1:7890 或 socks5://127.0.0.1:7891（根据你的代理）
PROXY_URL = os.getenv("PROXY_URL", "http://127.0.0.1:7890")

VOICE = "alloy"
PCM_SR = 16000
PORT = 5050

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/voice")
async def voice_bridge(ws: WebSocket):
    await ws.accept()
    print("Client connected")

    session = aiohttp.ClientSession()

    openai_ws = None

    try:
        # 连接 OpenAI Realtime (aiohttp 支持 proxy 参数)
        openai_ws = await session.ws_connect(
            OPENAI_WSS,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
            proxy=PROXY_URL,
            timeout=ClientWSTimeout(ws_close=60),
            receive_timeout=None,
        )
        print("Connected to OpenAI Realtime")

        # 初始化 session（告诉模型音频格式、voice等）
        init_msg = {
            "type": "session.update",
            "session": {
                "turn_detection": {"type": "server_vad"},
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": VOICE,
                #"modalities": ["text"],         # 如果只想用文本，不能只想用语音
                "modalities": ["audio","text"],  # 如果想用语音 + 文本
                "instructions": "You are a concise AI assistant.",
                "input_audio_transcription": {
                    "model": "whisper-1"
                }
            }
        }
        await openai_ws.send_str(json.dumps(init_msg))

        last_assistant_item = None
        openai_recv_task_cancel = asyncio.Event()

        async def from_client():
            """浏览器 -> OpenAI"""
            nonlocal last_assistant_item
            try:
                while True:
                    txt = await ws.receive_text()  # 浏览器发的是 JSON 文本
                    data = json.loads(txt)
                    # 期望 { "audio": "<base64 pcm16 bytes>" }
                    if "audio" in data:
                        b = base64.b64decode(data["audio"])
                        # send to OpenAI as base64 inside JSON
                        await openai_ws.send_str(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(b).decode("ascii")
                        }))
                    else:
                        # 可处理其他 message types，如 control messages
                        await openai_ws.send_str(json.dumps(data))
            except WebSocketDisconnect:
                raise
            except Exception as e:
                # 连接结束或其它错误
                print("from_client exception:", e)
                openai_recv_task_cancel.set()

        async def to_client():
            """OpenAI -> 浏览器（转发 response.audio.delta 等）"""
            nonlocal last_assistant_item
            try:
                async for msg in openai_ws:
                    # aiohttp WS 通过 .type 区分
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            j = json.loads(msg.data)
                        except Exception:
                            # 如果不是 JSON，则跳过
                            continue

                        mtype = j.get("type", "")
                        
                        # 【调试】如果收到用户转录结果，打印出来看看
                        if mtype == "conversation.item.input_audio_transcription.completed":
                            print("★ 用户语音转录:", j.get("transcript"))
                        
                        # 音频分片
                        if mtype == "response.audio.delta":
                            # delta 是 base64 字符串（OpenAI 返回的字段名可能是 'delta'）
                            pcm_b64 = j.get("delta") or j.get("audio") or j.get("audio_data")
                            if pcm_b64:
                                # print("收到 audio.delta，长度:", len(pcm_b64))
                                # 直接把 base64 字符串传给浏览器
                                await ws.send_text(json.dumps({
                                    "type": "response.audio.delta",
                                    "audio": pcm_b64
                                }))
                            last_assistant_item = j.get("item_id", last_assistant_item)

                        # 用户开始说话，打断 AI
                        elif mtype == "input_audio_buffer.speech_started":
                            if last_assistant_item:
                                # 发送 truncate 到 OpenAI
                                await openai_ws.send_str(json.dumps({
                                    "type": "conversation.item.truncate",
                                    "item_id": last_assistant_item,
                                    "content_index": 0,
                                    "audio_end_ms": 0
                                }))
                                last_assistant_item = None

                        else:
                            # 其余事件（比如 recognition.text.final / partial 等），转发或打印
                            await ws.send_text(json.dumps({"type": mtype, "data": j}))
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        # 罕见情况：直接二进制消息
                        # 直接 base64 编码后发给浏览器（不常用）
                        await ws.send_text(json.dumps({
                            "type": "binary",
                            "audio": base64.b64encode(msg.data).decode("ascii")
                        }))
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print("OpenAI ws error:", openai_ws.exception())
                        break
            except Exception as e:
                print("to_client exception:", e)
                openai_recv_task_cancel.set()

        # 并行执行
        tasks = [
            asyncio.create_task(from_client()),
            asyncio.create_task(to_client()),
        ]
        # 等待任一结束
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        # 取消未完成任务
        for t in pending:
            t.cancel()
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("Error:", e)
    finally:
        try:
            if openai_ws:
                await openai_ws.close()
        except Exception:
            pass
        await session.close()
        try:
            await ws.close()
        except Exception:
            pass
        print("Connection closed")

if __name__ == "__main__":
    import uvicorn
    print("Server listening on port", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
