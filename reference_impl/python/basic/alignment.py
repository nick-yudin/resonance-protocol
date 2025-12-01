import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine

# CONFIGURATION
# We need N >= D to fully constrain the rotation matrix.
# Since D=384, we use 1000 synthetic anchors for robust alignment.
DIMENSION = 384
NUM_SYNTHETIC_ANCHORS = 1000 

def simulate_alien_mind(vectors):
    """
    Simulates a different node by arbitrarily rotating the vector space.
    """
    dim = vectors.shape[1]
    # Generate a random rotation matrix (Orthogonal)
    H = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(H) 
    
    # Rotate the original world
    alien_vectors = np.dot(vectors, Q)
    return alien_vectors, Q

def align_worlds(source_anchors, target_anchors):
    """
    Calculates the Rotation Matrix R that best maps Source -> Target.
    """
    R, scale = orthogonal_procrustes(source_anchors, target_anchors)
    return R

def main():
    print("--- STEP 3: SEMANTIC ALIGNMENT (FIXED) ---")
    
    print("[1] Loading Core Logic...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # --- PHASE 1: CALIBRATION ---
    print(f"[2] Generating {NUM_SYNTHETIC_ANCHORS} synthetic vectors for calibration...")
    # Instead of dictionary words, we use mathematical noise to lock the geometry
    std_anchors = np.random.randn(NUM_SYNTHETIC_ANCHORS, DIMENSION)
    
    print("[3] Simulating Alien Node (Rotation)...")
    alien_anchors, true_rotation = simulate_alien_mind(std_anchors)
    
    dist_before = np.mean([cosine(std_anchors[i], alien_anchors[i]) for i in range(5)])
    print(f"    -> Average Distance before alignment: {dist_before:.4f} (Chaos)")

    print("[4] Calculating Procrustes Matrix (The Rosetta Stone)...")
    # Align Alien -> Standard
    R = align_worlds(alien_anchors, std_anchors)
    
    # --- PHASE 2: REAL WORLD TEST ---
    test_word = "The quick brown fox jumps over the lazy dog"
    print(f"\n[5] TESTING REAL SEMANTICS: '{test_word}'")
    
    # A. Standard View (Target)
    target_vector = model.encode([test_word])[0]
    
    # B. Alien View (Source) - We apply the SAME secret rotation to the real word
    source_vector_raw = np.dot(target_vector, true_rotation)
    
    # C. Attempt to Align
    # We use the Matrix R calculated from the noise vectors to decode the real word
    aligned_vector = np.dot(source_vector_raw, R)
    
    # D. Measure Error
    raw_dist = cosine(target_vector, source_vector_raw)
    aligned_dist = cosine(target_vector, aligned_vector)
    
    print(f"    RAW Transfer (No Alignment): Distance = {raw_dist:.4f}")
    print(f"    ALIGNED Transfer           : Distance = {aligned_dist:.8f}") # High precision
    
    # Precision threshold for float32 operations
    if aligned_dist < 1e-5:
        print("\n✅ SUCCESS: Perfect Semantic Telepathy.")
        print("   The Alien Node's vectors are now mathematically indistinguishable from ours.")
    else:
        print("\n❌ FAILED: Still drifting.")

if __name__ == "__main__":
    main()