from retriever import retriever
# Basic version of the NLP implementation, without the use of dependency parsing techniques.
import spacy
def nlp_graph_builder():
    nlp = spacy.load("en_core_web_md")

    query, chunks=retriever()

    # Filters the chunks from the previous code file, by testing the similarity match with the query, using the NLP library 'spaCy'.
    # It then returns the top k matching statements in the form of a list of sentences. 
    # k could be adjusted accordingly by the developers
    query_doc = nlp(query)
    sents_list = []

    for chunk in chunks:
        new_chunk = chunk.page_content.replace("\n"," ")
        doc=nlp(new_chunk)

        for sentence in doc.sents:
            score = sentence.similarity(query_doc)
            sents_list.append((score, sentence.text))

    sents_list.sort(key = lambda x: x[0], reverse = True)
    final_sents = sents_list[:30]

    # Process of Knowledge Graph building:

    # NLP processing
    sent_docs = [nlp(sent[1]) for sent in final_sents]

    # Function defined to normalize text, used ahead.
    def normalize(text : str):
        return text.strip().lower()

    def format(text : str):
        if " " in text:
            return text.capitalize()
        else:
            return text.upper()
    ents = {}
    for doc in sent_docs:
        for ent in doc.ents:
            ents.add(normalize(ent.text))

    node_ents = []

    for ent in ents:
        node_ents.append(format(ent))

    for i in range(len(node_ents)-1):
        print(node_ents[i]+" :")
        for j in range(i+1, len(node_ents)):
            print("|------> co-occurs : "+node_ents[j])
    return 