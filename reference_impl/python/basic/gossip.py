import uuid
import random
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# --- CONFIGURATION ---
NUM_NODES = 10
CONNECTION_PROBABILITY = 0.3
SEMANTIC_THRESHOLD = 0.35     # Sensitivity to meaning (Raised to filter synonyms)
TTL_LIMIT = 3

class SemanticEvent:
    def __init__(self, content, vector, source_id, ttl):
        self.id = str(uuid.uuid4())[:8]
        self.content = content
        self.vector = vector
        self.source_id = source_id
        self.ttl = ttl
        self.trace = []

class MeshNode:
    def __init__(self, node_id, model):
        self.node_id = f"NODE_{node_id:02d}"
        self.model = model
        self.peers = []
        self.memory = set()
        self.last_vector = None

    def connect(self, other_node):
        if other_node not in self.peers:
            self.peers.append(other_node)
            other_node.peers.append(self)

    def inject_thought(self, text):
        print(f"\n💉 INJECTION @ {self.node_id}: '{text}'")
        vector = self.model.encode(text)
        event = SemanticEvent(text, vector, self.node_id, TTL_LIMIT)
        self.process_event(event)

    def process_event(self, event):
        # 1. Deduplication
        if event.id in self.memory:
            return
        self.memory.add(event.id)
        event.trace.append(self.node_id)

        # 2. Semantic Filtering
        impact = 0.0
        if self.last_vector is not None:
            impact = cosine(self.last_vector, event.vector)
            if impact < SEMANTIC_THRESHOLD:
                return # Silence (Energy Saved)
        
        self.last_vector = event.vector
        
        # 3. Visualization
        indent = "  " * (TTL_LIMIT - event.ttl)
        print(f"{indent}⚡ {self.node_id} received '{event.content}' (Impact: {impact:.3f} | TTL: {event.ttl})")

        # 4. Propagation (Gossip)
        if event.ttl > 0:
            event.ttl -= 1
            for peer in self.peers:
                peer.process_event(event)
        else:
            print(f"{indent}  X Energy dissipated at {self.node_id}.")

def create_network():
    print("--- STEP 4: GOSSIP PROTOCOL SIMULATION ---")
    print("[1] Loading Core Logic...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    nodes = [MeshNode(i, model) for i in range(NUM_NODES)]
    
    print(f"[2] Wiring {NUM_NODES} nodes into a Mesh...")
    # Basic line backbone
    for i in range(NUM_NODES - 1):
        nodes[i].connect(nodes[i+1])
    
    # Random connections
    for i in range(NUM_NODES):
        if random.random() < CONNECTION_PROBABILITY:
            target = random.choice(nodes)
            if target != nodes[i]:
                nodes[i].connect(target)

    print("\n--- NETWORK TOPOLOGY ---")
    for n in nodes:
        peer_ids = [p.node_id for p in n.peers]
        print(f"{n.node_id} <-> {peer_ids}")
    
    return nodes

if __name__ == "__main__":
    network = create_network()
    
    print("\n--- PHASE 1: INITIALIZATION ---")
    network[0].inject_thought("System is online and waiting.")
    
    print("\n--- PHASE 2: NOISE TEST (Synonyms) ---")
    network[0].inject_thought("The system is ready and standby.") 
    
    print("\n--- PHASE 3: EVENT CASCADE (The Wave) ---")
    network[0].inject_thought("WARNING: HOSTILE DRONE DETECTED!")