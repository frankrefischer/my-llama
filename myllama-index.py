from typing import List

import click
from pathlib import Path

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document


@click.command()
def main():
    # define what documents to load
    txt_input_dir: Path = Path('~/myllama/txts').expanduser().absolute()
    loader = DirectoryLoader(path=txt_input_dir.as_posix(),
                             loader_cls=TextLoader,
                             silent_errors=True)
    print(f'loading documents from: {txt_input_dir.as_posix()}')
    documents: List[Document] = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                              chunk_overlap=50)
    print(f'splitting {len(documents)} documents')
    texts: List[Document] = splitter.split_documents(documents)
    print(f'splitted into {len(texts)} texts')

    print(f'loading embeddings')
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    print(f'vectorizing texts')
    db = FAISS.from_documents(texts, embeddings)
    faiss_db_dir = Path('~/myllama').expanduser().absolute()
    print(f'saving faiss db to: {faiss_db_dir}')
    db.save_local(folder_path=faiss_db_dir.as_posix())
    print(f'finished')


if __name__ == '__main__':
    main()