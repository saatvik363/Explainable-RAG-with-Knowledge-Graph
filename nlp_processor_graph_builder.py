import spacy
nlp = spacy.load("en_core_web_md")

from collections import defaultdict

query_doc=nlp(query)

sentence_list=[]

for chunk in chunks:
    new_chunk=chunk.page_content.replace("\n"," ")

    doc=nlp(new_chunk)

    for sentence in doc.sents:
        score=sentence.similarity(query_doc)

        sentence_list.append((score, sentence.text))

sentence_list.sort(key=lambda x: x[0], reverse=True)

final_sentences=sentence_list[:5]

# Formatting the entities for output.
def format_entity(entity_text: str) -> str:
    entity_text = entity_text.strip()
    
    if " " not in entity_text:
        return entity_text.upper()
    else:
        return entity_text.title()

# Fix: extract only sentence text, not (score, text) tuple
docs = [nlp(sent[1]) for sent in final_sentences]


# Create a set of normalized entities.
entities = set()
for doc in docs:
    for entity in doc.ents:
            normalized = entity.text.strip().lower()
            entities.add(normalized)

# Create a formatted list for output.
f_entities = [format_entity(ent) for ent in entities]

# Knowledge Graph data structure
# graph[source_entity] = list of (relation, target_entity)
graph = defaultdict(list)

# Sentence-wise processing
for doc in docs:
    for sent in doc.sents:

        # collect entities present in this sentence
        sent_entities = []
        for ent in doc.ents:
            if ent.start >= sent.start and ent.end <= sent.end:
                sent_entities.append(ent)

        # Normalize sentence entities
        norm_sent_entities = {
            ent.text.strip().lower(): ent for ent in sent_entities
        }

        # dependency-based relations
        found_relation = False

        for token in sent:
            if token.pos_ == "VERB":
                subjects = []
                objects = []

                # Inspect dependency children
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subjects.append(child)
                    elif child.dep_ in ("dobj", "pobj"):
                        objects.append(child)

                # Create relations if both sides exist
                for subj in subjects:
                    for obj in objects:
                        subj_text = subj.text.strip().lower()
                        obj_text = obj.text.strip().lower()

                        if subj_text in entities and obj_text in entities:
                            relation = token.lemma_
                            graph[subj_text].append((relation, obj_text))
                            found_relation = True

        # Fall-back to co-occurrence if no verb relation found
        if not found_relation:
            keys = list(norm_sent_entities.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    graph[keys[i]].append(("co-occurs", keys[j]))

# Output Knowledge Graph
print("\nKnowledge Graph:\n")

for entity in entities:
    print(f"{format_entity(entity)}:")
    if entity in graph:
        for rel, target in graph[entity]:
            print(f"  {rel} -> {format_entity(target)}")
    else:
        print("  (no outgoing relations)")
