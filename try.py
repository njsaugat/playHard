from gliner2 import GLiNER2
import requests

# Load model once, use everywhere
extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")


url= "https://storage.googleapis.com/core-production-3c790.appspot.com/transcripts/RSS-https://api.spreaker.com/episode/68290761-1763549167632.json"
resp = requests.get(url, timeout=10)
count = 0
totalWords = 0
resp.raise_for_status()          # raises if 4xx/5xx
data = resp.json()               # parsed JSON (dict/list)

transcriptByWords = data["transcriptByWords"]
# Extract entities in one line
# text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."

totalWords = 0
count = 0

for transcript in transcriptByWords:
    # Ensure transcript is a dict
    if not isinstance(transcript, dict):
        continue

    # Safely get words
    transcrpt = transcript.get("words")
    if not transcrpt:
        continue

    # Run extraction
    result = extractor.extract_entities(transcrpt, ["company", "product"])

    # Ensure result structure is valid
    if not isinstance(result, dict):
        continue

    entities = result.get("entities")
    if not isinstance(entities, dict):
        continue

    companies = entities.get("company", [])
    products = entities.get("product", [])

    # Ensure both are lists
    if not isinstance(companies, list) or not isinstance(products, list):
        continue

    # Skip if nothing found
    if not companies or not products:
        continue

    totalWords += len(transcrpt)
    count += 1
    print(result)

print(totalWords, count)



# print(result)
# {'entities': {'company': ['Apple'], 'person': ['Tim Cook'], 'product': ['iPhone 15'], 'location': ['Cupertino']}}
