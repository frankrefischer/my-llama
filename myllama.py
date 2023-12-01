"""
This script reads the database of information from local text files
and uses a large language model to answer questions about their content.
"""

from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from pathlib import Path


def prepare():
    print('loading language model')
    model_file = Path('~/myllama/llama-2-7b-chat.ggmlv3.q8_0.bin').expanduser().absolute()
    llm = CTransformers(model=model_file.as_posix(),
                        model_type='llama',
                        config={'max_new_tokens': 256, 'temperature': 0.01})

    print('loading embeddings')
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})
    print('loading faiss database')
    faiss_db_dir = Path('~/myllama').expanduser().absolute()
    db = FAISS.load_local(faiss_db_dir.as_posix(), embeddings)

    print('preparing embeddings')
    # prepare llm pre-loaded with faiss db content
    retriever = db.as_retriever(search_kwargs={'k': 2})

    # prepare the template we will use when prompting the AI
    template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question'])
    qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                         chain_type='stuff',
                                         retriever=retriever,
                                         return_source_documents=True,
                                         chain_type_kwargs={'prompt': prompt})

    def q(prompt: str, only_result: bool = True):
        output = qa_llm({'query': prompt})
        a: str = output["result"]
        print(a.replace('\\n', '\n'))
        if not only_result:
            return output

    return q

