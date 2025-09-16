import threading
import socket
import time
import json
import traceback

from Retrieval import Retrieval

HOST = "localhost"
PORT = 11452
HEADER = 4
vector_db       = "./vectorDB/test.db"
collection_name = "collection_test"
embedding_model = "Qwen/Qwen3-Embedding-0.6B"


def recv(client : socket.socket):
    recv_size = int.from_bytes(client.recv(HEADER))
    recv_msg  = client.recv(recv_size).decode("utf-8")
    return recv_msg, recv_size

def send(client : socket.socket, send_msg : str):
    send_bytes = send_msg.encode("utf-8")
    send_size = int.to_bytes(len(send_bytes), length = HEADER)
    client.send(send_size)
    client.send(send_bytes)
    return len(send_bytes)
    

def handle_client(client : socket.socket, address, func):
    recv_msg, recv_size = recv(client)
    print(f"Receive message from {address}: {recv_msg}  .Size = {recv_size}")

    results = {}
    for i, result in enumerate(func([recv_msg], limit = 20)):
        results[i] = result
    
    send_msg = json.dumps(
        obj = results, 
        ensure_ascii = False
        )

    print("Request finished.")

    try:
        send_size = send(client, send_msg)
        print(f"Send message to {address}. Size= {send_size}")
    except Exception as e:
        print(e.__traceback__)
    finally:
        client.close()
        


def start_server():
    server = socket.socket(
        socket.AF_INET, 
        socket.SOCK_STREAM,
    )
    server.bind((HOST, PORT))
    server.listen()
    print("Server started:", server)

    retrival = Retrieval(
        vector_db       = vector_db,
        collection_name = collection_name,
        embedding_model = embedding_model
    )
    retrival.load()
    print("Retrieval service loaded")

    while True:
        client, address = server.accept()
        thread = threading.Thread(
            target = handle_client,
            args = (client, address, retrival.search_by_query))
        thread.start()
        

if __name__ == "__main__" :

    while True:
        try:
            start_server()
        except Exception as e:
            print(e.__traceback__)
            print("Some problems occured, server will restart in 5 seconds.")
            time.sleep(5)
            start_server()