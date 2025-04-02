import os

import openai

OPENAI_TEMP = 0.5
OPENAI_DEDUPLICATE_THRESHOLD = 0.9
OPENAI_MODEL = "gpt-3.5-turbo"

with open("/u/atharvas/.keys/api_utaustin.key", "r") as f:
    api_key = f.read()

if os.path.exists("/u/atharvas/.keys/organization.key"):
    with open("/u/atharvas/.keys/organization.key", "r") as f:
        organization_key = f.read()
    organization_key = organization_key.strip()
else:
    organization_key = None

openai_client = openai.OpenAI(api_key=api_key, organization=organization_key)
vllm_client = openai.OpenAI(
    api_key="token-abc123", base_url="http://localhost:11440/v1"
)
