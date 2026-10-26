# ==================== SETING UP ENVIRONMENTS ====================
# Proxy and Key settings  -----------------------------------------------------
import os
from dotenv import load_dotenv
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
# CRITICAL: Bypass the proxy for local Tempo traffic
os.environ['no_proxy'] = "localhost,127.0.0.1"
os.environ['NO_PROXY'] = "localhost,127.0.0.1"
os.chdir("/home/ubuntu/Building_Applications_with_AI_Agents")
load_dotenv()

import warnings
# Suppress the Pydantic serialization noise caused by OpenAI 2.x
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

import dspy 
# New way (DSPy 3.x)
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
dspy.settings.configure(cache=False) # No cache

def lookup_threat_intel(indicator: str) -> str:
  """Mock: Look up threat intelligence for an indicator.""" 
  return f"Mock intel for {indicator}: potentially malicious"

def query_logs(query: str) -> str:
  """Mock: Search and analyze security logs."""
  return f"Mock logs for '{query}': suspicious activity detected"

# Handful of synthetic test cases (alert -> expected answer) 
# In practice, derive from real logs or annotate failures;
# aim for 100+ for better optimization 
trainset = [
  dspy.Example(alert='''Suspicious login attempt 
              from IP 203.0.113.45 to admin account.''',
              answer='''Lookup threat intel for IP, query logs for activity, 
                triage as true positive, isolate host if malicious.''')
        .with_inputs('alert'),
  dspy.Example(alert='''Unusual file download from 
              URL example.com/malware.exe.''',
              answer='''Lookup threat intel for URL and hash,
                query logs for endpoint activity, triage as true positive, 
                isolate host.''').with_inputs('alert'),
  dspy.Example(alert='''High network traffic to domain suspicious-site.net.''', 
              answer='''Lookup threat intel for domain, query logs for
                network and firewall, triage as false positive if benign.''')
        .with_inputs('alert'),
  dspy.Example(alert='''Alert: Potential phishing email 
              with attachment hash abc123.''',
              answer='''Lookup threat intel for hash, query logs 
                for email and endpoint, triage as true positive, 
                send analyst response.''').with_inputs('alert'),
  dspy.Example(alert='''Anomaly in user behavior: multiple failed 
              logins from new device.''',
              answer='''Query logs for authentication, 
                lookup threat intel for device IP, 
                triage as true positive if pattern matches attack.''')
        .with_inputs('alert'),
]

# 1. Define flexible metric, just for demo
# The previouse dspy.evaluate.answer_exact_match will returns score 0
# if the answer is not exactly same as the expected one
def soc_metric(example, pred, trace=None):
    keywords = ["intel", "logs", "triage", "malicious", "ip"]
    found = sum(1 for k in keywords if k in pred.answer.lower())
    return found / len(keywords)

from sentence_transformers import SentenceTransformer, util

# Load a lightweight, fast model for evaluation
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity_metric(gold, pred, trace=None):
    gold_text = gold.answer
    pred_text = pred.answer
    
    # Encode the texts into embeddings
    gold_emb = similarity_model.encode(gold_text, convert_to_tensor=True)
    pred_emb = similarity_model.encode(pred_text, convert_to_tensor=True)
    
    # Calculate Cosine Similarity
    score = util.pytorch_cos_sim(gold_emb, pred_emb).item()
    
    # Optional: Set a floor. If similarity is below 0.5, it's probably wrong.
    return max(0, score)


# Define ReAct module for SOC incident handling
react = dspy.ReAct("alert -> answer", tools=[lookup_threat_intel, query_logs])
# Optimizer with a simple metric 
# (exact match for illustration; 
# use a more nuanced metric like
# semantic similarity in production)
# tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
# optimized_react = tp.compile(react, trainset=trainset)
# semantic_similarity_metric, soc_metric
# minibatch=False if examples<35
tp = dspy.MIPROv2(metric=semantic_similarity_metric, auto=None, num_threads=24, num_candidates=3) 
optimized_react = tp.compile(react, trainset=trainset, num_trials=1, minibatch=False)

# See what the 'optimized' agent actually said
example = trainset[0]
res = optimized_react(alert=example.alert)
print(f"Prediction: {res.answer}")
print(f"Expected: {example.answer}")

# Inspect the actual prompt to see if the tools are being called correctly
dspy.inspect_history(n=1)
