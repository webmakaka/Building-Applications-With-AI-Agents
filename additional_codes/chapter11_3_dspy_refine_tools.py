# ==================== SETTING UP ENVIRONMENTS ====================
import os
import warnings
import dspy
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Proxy and Key settings
os.environ['http_proxy'] = "http://127.0.0.1:7890"
os.environ['https_proxy'] = "http://127.0.0.1:7890"
os.environ['no_proxy'] = "localhost,127.0.0.1"
os.environ['NO_PROXY'] = "localhost,127.0.0.1"
os.chdir("/home/ubuntu/Building_Applications_with_AI_Agents")
load_dotenv()

# Suppress Pydantic noise
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# DSPy 3.x Configuration
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
dspy.settings.configure(cache=False)

# --- 1. Define Signatures and Modules ---

class ThreatClassifier(dspy.Signature):
    """Classify the threat level of a given indicator (e.g., IP, URL, hash) as 
    'benign', 'suspicious', or 'malicious'."""
    indicator: str = dspy.InputField(desc="The indicator to classify, such as an IP address, URL, or file hash.")
    threat_level: str = dspy.OutputField(desc="The classified threat level: 'benign', 'suspicious', or 'malicious'.")

class ThreatClassificationModule(dspy.Module):
    def __init__(self): 
        super().__init__()
        # Using ChainOfThought to allow the model to reason before classifying
        self.classify = dspy.ChainOfThought(ThreatClassifier)
  
    def forward(self, indicator):
        return self.classify(indicator=indicator)

# --- 2. Dataset Setup ---

trainset = [
    dspy.Example(indicator="203.0.113.45", threat_level="suspicious").with_inputs('indicator'), # Known malicious IP
    dspy.Example(indicator="example.com/malware.exe", threat_level="malicious").with_inputs('indicator'), # Malicious URL
    dspy.Example(indicator="benign-site.net", threat_level="benign").with_inputs('indicator'), # Safe domain
    dspy.Example(indicator="abc123def456", threat_level="malicious").with_inputs('indicator'), # Malware hash
    dspy.Example(indicator="192.168.1.1", threat_level="benign").with_inputs('indicator'), # Local IP
    dspy.Example(indicator="obfuscated.url/with?params", threat_level="suspicious").with_inputs('indicator'), # Edge case: obfuscated URL
    dspy.Example(indicator="new-attack-vector-hash789", threat_level="malicious").with_inputs('indicator'), # Novel threat
]

# --- 3. Advanced Metrics (from chapter 11.2) ---

# Load a lightweight model for semantic evaluation
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_threat_metric(gold, pred, trace=None):
    """Evaluates how close the predicted threat level is to the gold label using embeddings."""
    gold_text = gold.threat_level
    pred_text = pred.threat_level
    
    # Simple Exact Match check first for classification tasks
    if gold_text.lower() == pred_text.lower():
        return 1.0
    
    # Fallback to Semantic Similarity
    gold_emb = similarity_model.encode(gold_text, convert_to_tensor=True)
    pred_emb = similarity_model.encode(pred_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(gold_emb, pred_emb).item()
    
    return max(0, score)

# --- 4. Optimization (Upgraded to MIPROv2) ---

# We use MIPROv2 to refine the prompt instructions and selection of few-shot examples
# auto=None for manual control, or auto="light" for automated prompt engineering
tp = dspy.MIPROv2(metric=semantic_threat_metric, auto=None, num_threads=24, num_candidates=3)

# Compile the module
# minibatch=False is used because our trainset is small (< 35 examples)
optimized_module = tp.compile(
    ThreatClassificationModule(), 
    trainset=trainset, 
    num_trials=5, 
    minibatch=False
)

# --- 5. Usage and Inspection ---

def classify_threat(indicator: str) -> str:
    """Classify threat level using the optimized DSPy module.""" 
    prediction = optimized_module(indicator=indicator)
    return prediction.threat_level

# Test Case
test_indicator = "10.0.0.5"
result = classify_threat(test_indicator)

print(f"--- Optimization Result ---")
print(f"Indicator: {test_indicator}")
print(f"Predicted Threat Level: {result}")

# Inspect the refined prompt and reasoning
print("\n--- Refined Prompt History ---")
dspy.inspect_history(n=1)
