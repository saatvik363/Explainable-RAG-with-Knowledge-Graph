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