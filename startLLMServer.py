import asyncio
import socket
import time
import json
import traceback

from Language_Model import Language_Model as LLM
from Language_Model import messages_generator

HOST        = "localhost"
PORT        = 11451
RAGHOST     = "localhost"
RAGPORT     = 11452
HEADER      = 4
model_name  = "Qwen/Qwen3-0.6B"
temperature = 0.4

async def recv(reader : asyncio.StreamReader):
    recv_size = int.from_bytes(await reader.read(HEADER))
    recv_msg = (await reader.read(recv_size)).decode("utf-8")
    return recv_msg, recv_size
    
async def send(writer : asyncio.StreamWriter, msg : str):
    send_bytes = msg.encode("utf-8")
    send_size = int.to_bytes(len(send_bytes), length = HEADER)
    writer.write(send_size)
    writer.write(send_bytes)
    await writer.drain()
    return len(send_bytes)

async def handle_client(
        reader : asyncio.StreamReader,
        writer : asyncio.StreamWriter,
        llm    : LLM
        ):
    addr = writer.get_extra_info(name = "peername")

    recv_msg, recv_size = await recv(reader)
    print(f"Receive message from {addr}: {recv_msg}  .Size = {recv_size}")

    json_obj = json.loads(recv_msg)
    query = str(json_obj["query"])
    enable_rag = bool(json_obj["enable_rag"])
    enable_thinking = bool(json_obj["enable_thinking"])
    retrieval_results = None
    if enable_rag:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((RAGHOST, RAGPORT))
        client.send(int.to_bytes(recv_size, HEADER))
        client.send(query.encode("utf-8"))

        retrieval_size = int.from_bytes(client.recv(HEADER))
        retrieval_results = client.recv(retrieval_size).decode("utf-8")
        retrieval_results = json.loads(retrieval_results)

    messages = messages_generator(query, retrieval_results)

    print("Query and retrival results: ", messages)
    answer, consumed_tokens = llm.generate(messages, enable_thinking = enable_thinking)
    send_msg = json.dumps(
        obj = {"answer": answer, "usage": consumed_tokens}, 
        ensure_ascii = False
        )

    print("Request finished.")

    try:
        send_size = await send(writer, send_msg)
        print(f"Send message to {addr}. Size= {send_size}")
    except Exception as e:
        print(traceback.format_tb(e.__traceback__))
    finally:
        writer.close()

    print()

async def start_server():
    llm = LLM(
        model_name      = model_name ,
        temperature     = temperature
    )
    llm.load()

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, llm),
        HOST,
        PORT
        )
    print("Server started:", server.sockets)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(start_server())