loader = DirectoryLoader('./news', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()
text_splitter = CharacterTextSplitter()
chunks = text_splitter.split_documents(documents)
db = Chroma.from_documents(chunks, embeddings_client)
retriever = db.as_retriever()

retrieval_augmented_qa_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": prompt | llm, "context": itemgetter("context")}
)
answers = []
contexts = []

for query in questions:
    print(query)
    response = retrieval_augmented_qa_chain.invoke({"question": query})

    # Access the response content
    answers.append(response["response"].content)

    # Access the context content
    context_content = [context.page_content for context in response["context"]]
    contexts.append(context_content)

result = evaluation_rag(questions, answers, contexts, ground_truths)
print(result)
df = result.to_pandas()
df