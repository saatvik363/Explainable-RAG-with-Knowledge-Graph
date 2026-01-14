#STEP1:Prepare Retrieved Context
retrieved_chunks=[]

if not results:
    raise ValueError("No documents retrieved for the query.")

for idx, doc in enumerate(results):
    retrieved_chunks.append({
        "chunk_id":f"C{idx+1}",
        "text":doc.page_content.strip()
    })

#STEP2:Prompt Assembly
def build_prompt(query, chunks):
    context_block=""
    for chunk in chunks:
        context_block+=f"[{chunk['chunk_id']}]\n{chunk['text']}\n\n" #appends each chunk to context_block

    #Start a multi-line formatted string which is to be passed as a prompt to the LLM
    prompt=f"""
You are an assistant answering a question using ONLY the provided text snippets.

Question:
{query}

Retrieved Text Chunks:
{context_block}

Instructions:
1.Answer using ONLY the retrieved text.
2.Do not introduce external knowledge.
3.Cite chunk IDs (e.g., C1, C2) for every factual claim.
4.If the answer is not present, say:
    "The retrieved text does not contain sufficient information."

Output strict JSON:
{{
  "answer": "...",
  "supporting_chunks": ["C1", "C2"]
}}
"""
    return prompt

#STEP3:Call LLaMA-3 via Ollama
import requests     #Used to make HTTP calls
import json     #Imports python JSON module

OLLAMA_URL="http://localhost:11434/api/generate"

prompt=build_prompt(query, retrieved_chunks)

payload={
        "model":"llama3:8b",
        "prompt":prompt,
        "stream":False
        }       #This dictionary becomes the HTTP request body

response=requests.post(OLLAMA_URL, json=payload, timeout=120)

#Check if the request has failed
if response.status_code!=200:
    raise RuntimeError("Ollama request failed")

response_json=response.json()

if "response" not in response_json:
    raise RuntimeError("Ollama response missing 'response' field")

raw_output=response_json["response"]      #Extracts generated text from Ollama response

#STEP4:Parse Output
try:
    parsed_output=json.loads(raw_output)    #Parse raw_output into python dictionary
except json.JSONDecodeError:
    raise ValueError("LLM output is not valid JSON")

answer=parsed_output.get("answer", "")      #Extracts answer from parsed JSON

supporting_chunks=parsed_output.get("supporting_chunks", [])

valid_ids={c["chunk_id"] for c in retrieved_chunks}
for cid in supporting_chunks:
    if cid not in valid_ids:
        raise ValueError(f"Invalid chunk ID cited:{cid}")

print("\nFINAL ANSWER:\n")
print(answer)

print("\nSUPPORTED BY CHUNKS:\n")
print(", ".join(supporting_chunks))