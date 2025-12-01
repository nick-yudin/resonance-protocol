import socket
import struct
import resonance_pb2

# Network Configuration
HOST = '0.0.0.0'
PORT = 5000

def recvall(sock, n):
    """Helper function to ensure we receive exactly n bytes."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def start_receiver():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allow port reuse to avoid 'Address already in use' errors during testing
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"[RECEIVER] Listening strictly for Meaning on port {PORT}...")
        print("[RECEIVER] Waiting for a semantic connection...")

        conn, addr = server.accept()
        print(f"[RECEIVER] Link established with {addr}")

        with conn:
            while True:
                # 1. Read the message length (4 bytes, big-endian)
                raw_len = recvall(conn, 4)
                if not raw_len: 
                    print("[RECEIVER] Connection closed by sender.")
                    break
                
                msg_len = struct.unpack('>I', raw_len)[0]

                # 2. Read the actual payload based on the length
                data = recvall(conn, msg_len)
                if not data: 
                    break

                # 3. Deserialize (Protobuf -> Object)
                event = resonance_pb2.SemanticEvent()
                event.ParseFromString(data)

                # Visualize the received "Meaning"
                vec_preview = event.embedding[:3] # Show first 3 dims
                print(f"📥 EVENT RECEIVED | Source: {event.source_id}")
                print(f"   Context: '{event.debug_label}'")
                print(f"   Vector: {vec_preview}... [dim={len(event.embedding)}]")
                print("-" * 50)

    except KeyboardInterrupt:
        print("\n[RECEIVER] Shutting down.")
    finally:
        server.close()

if __name__ == "__main__":
    start_receiver()