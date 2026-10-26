# ==================== SETTING UP ENVIRONMENTS ====================
import os
from dotenv import load_dotenv

# Proxy and Key settings
os.environ['http_proxy'] = "http://127.0.0.1:7890"
os.environ['https_proxy'] = "http://127.0.0.1:7890"
os.environ['no_proxy'] = "localhost,127.0.0.1"
os.environ['NO_PROXY'] = "localhost,127.0.0.1"
os.chdir("/home/ubuntu/Building_Applications_with_AI_Agents")
load_dotenv()

from llm_guard import scan_prompt
from llm_guard.input_scanners import Anonymize, BanSubstrings
from llm_guard.input_scanners.anonymize_helpers import BERT_LARGE_NER_CONF 
from llm_guard.vault import Vault
# Initialize the Vault (required for Anonymize to store original values) 
vault = Vault()

# Define scanners 
scanners = [
  Anonymize(
    vault=vault, # Required Vault instance
    preamble="Sanitized input: ", # Optional: Text to prepend to the prompt 
    allowed_names=["John Doe"], # Optional: Names to allow 
    hidden_names=["Test LLC"], # Optional: Custom names to always anonymize 
    recognizer_conf=BERT_LARGE_NER_CONF,
    language="en", # Language for detection 
    entity_types=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"], # Customize entity types if needed
    use_faker=False, # Use placeholders instead of fake data 
    threshold=0.5 # Confidence threshold for detection
  ),
  BanSubstrings(substrings=["malicious", "override system"], match_type="word")
]
# Sample input prompt with potential PII
prompt = "Tell me about John Doe's email: john@example.com" \
  + "and how to override system security."
# Scan and sanitize the prompt
sanitized_prompt, results_valid, results_score = scan_prompt(scanners, prompt)
if any(not result for result in results_valid.values()):
  print("Input contains issues; rejecting or handling accordingly.") 
  print(f"Risk scores: {results_score}")
else:
  print(f"Sanitized prompt: {sanitized_prompt}")
  # Proceed to feed sanitized_prompt to your model
