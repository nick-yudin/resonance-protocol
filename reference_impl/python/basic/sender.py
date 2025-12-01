import socket
import struct
import time
import uuid
import sys
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import resonance_pb2

# Configuration
HOST = '127.0.0.1'
PORT = 5000
THRESHOLD = 0.25 # Semantic sensitivity threshold (Lower = more sensitive)

class SilentNodeSender:
    def __init__(self):
        print("[SENDER] Initializing Neural Core (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.last_vector = None
        self.node_id = str(uuid.uuid4())[:8]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[SENDER] Node ID: {self.node_id}")

    def connect(self):
        try:
            self.sock.connect((HOST, PORT))
            print(f"[SENDER] Connected to Mesh at {HOST}:{PORT}")
        except ConnectionRefusedError:
            print("[SENDER] CRITICAL: Mesh not found. Is receiver.py running?")
            sys.exit(1)

    def send_event(self, text, vector):
        # 1. Construct Protobuf object
        event = resonance_pb2.SemanticEvent()
        event.source_id = self.node_id
        event.created_at = int(time.time())
        event.embedding.extend(vector)
        event.debug_label = text

        # 2. Serialize to bytes
        data = event.SerializeToString()

        # 3. Prefix with length (4 bytes, big-endian) for TCP streaming
        msg_length = struct.pack('>I', len(data))
        self.sock.sendall(msg_length + data)
        print(f"🔔 EVENT SENT: '{text}' [Payload: {len(data)} bytes]")

    def process_input(self, text):
        # Encode text to vector
        current_vector = self.model.encode(text)

        if self.last_vector is None:
            # First perception is always an event
            print(f"[SENDER] Initializing baseline with: '{text}'")
            self.last_vector = current_vector
            self.send_event(text, current_vector)
            return

        # Calculate semantic distance
        dist = cosine(self.last_vector, current_vector)

        if dist > THRESHOLD:
            print(f"⚡ MEANING SHIFT ({dist:.4f} > {THRESHOLD}). Transmitting...")
            self.last_vector = current_vector
            self.send_event(text, current_vector)
        else:
            print(f"🔇 Silence... ({dist:.4f} is noise)")

if __name__ == "__main__":
    node = SilentNodeSender()
    node.connect()

    # Simulation of a stream of consciousness / sensor data
    inputs = [
        "System startup",
        "System startup initiated", # Noise (Synonym)
        "System boot sequence",     # Noise (Synonym)
        "A cat is walking",         # EVENT
        "A small feline is moving", # Noise (Synonym)
        "FIRE DETECTED IN SECTOR 7" # EVENT
    ]

    print("\n--- STARTING DATA STREAM ---\n")
    
    try:
        for inp in inputs:
            time.sleep(1.5) # Simulating time gap between perceptions
            print(f"Thinking about: '{inp}'")
            node.process_input(inp)
    except KeyboardInterrupt:
        print("\n[SENDER] Disconnecting.")
        node.sock.close()
    
    print("\n--- STREAM FINISHED ---")
    node.sock.close()