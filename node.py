"""
RESONANCE PROTOCOL // LEVEL 1 REFERENCE IMPLEMENTATION
Node Emulator v1.2 (Protobuf Integrated)

Core Logic:
1. Silence is Default: Emit event only if cosine_distance > threshold.
2. Semantic Delta: Emit only the difference vector (delta), not the state.
3. Wire Format: Uses Google Protocol Buffers for serialization.
"""

from __future__ import annotations
import time
import numpy as np
import uuid
from typing import Optional
from sentence_transformers import SentenceTransformer

# Import generated Protobuf code
# (Make sure you ran: python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. resonance.proto)
try:
    import resonance_pb2
except ImportError:
    print("ERROR: resonance_pb2 not found. Did you compile the proto file?")
    print("Run: python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. resonance.proto")
    exit(1)

# --- CONFIGURATION ---
MODEL_NAME = 'all-MiniLM-L6-v2'
DEFAULT_THRESHOLD = 0.45

class ResonanceNode:
    def __init__(self, node_id: str, threshold: float = DEFAULT_THRESHOLD):
        self.node_id = node_id
        self.threshold = threshold
        
        print(f"[{self.node_id}] Loading Neural Core ({MODEL_NAME})...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.current_embedding: Optional[np.ndarray] = None
        
        # Get model dimension (usually 384 for MiniLM)
        self.dim = self.model.get_sentence_embedding_dimension()
        
        self.stats = {"events": 0, "silence": 0, "total": 0}
        print(f"[{self.node_id}] Online. Dim: {self.dim}. Threshold: {self.threshold}")

    def process(self, text: str) -> Optional[resonance_pb2.SemanticEvent]:
        """
        Main Loop: Input -> Embedding -> Cosine Check -> Proto Event/Silence
        """
        self.stats["total"] += 1
        
        # 1. Perception
        new_embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

        # 2. Genesis
        if self.current_embedding is None:
            return self._emit_proto_event(new_embedding, new_embedding, text)

        # 3. Cognition
        distance = 1.0 - np.dot(self.current_embedding, new_embedding)
        
        # 4. Resonance Check
        if distance > self.threshold:
            delta = new_embedding - self.current_embedding
            return self._emit_proto_event(new_embedding, delta, text)
        else:
            self.stats["silence"] += 1
            return None

    def _emit_proto_event(self, new_state, delta, text) -> resonance_pb2.SemanticEvent:
        self.current_embedding = new_state
        self.stats["events"] += 1
        
        # Create Protobuf object
        event = resonance_pb2.SemanticEvent()
        event.event_id = uuid.uuid4().bytes # 16 bytes UUID
        event.timestamp_us = int(time.time() * 1_000_000)
        event.node_id = self.node_id
        event.embedding_dim = self.dim
        event.semantic_domain = "text"
        event.confidence = 1.0
        event.debug_text = text
        
        # Write vector (float array)
        event.delta_vector.extend(delta.tolist())
        
        return event

    def print_stats(self):
        ratio = self.stats['silence'] / self.stats['total'] if self.stats['total'] > 0 else 0
        print(f"\n📊 NODE STATS: Compression Ratio {ratio:.1%}")
        print(f"   Total: {self.stats['total']} | Events: {self.stats['events']} | Silence: {self.stats['silence']}")

# --- MAIN DEMO ---
if __name__ == "__main__":
    node = ResonanceNode("NODE_01")
    
    stream = [
        "System startup",
        "System startup initiated",
        "Cat is walking",
        "A small cat is moving",
        "FIRE!",
    ]

    print("\n--- PROTOBUF STREAM STARTED ---")
    for msg in stream:
        event_proto = node.process(msg)
        if event_proto:
            # Serialize to bytes (as for network)
            binary_data = event_proto.SerializeToString()
            print(f"🔔 EVENT [Size: {len(binary_data)} bytes] -> {msg}")
        else:
            print(f"🔇 ... silence ... ('{msg}')")
        time.sleep(0.1)

    node.print_stats()