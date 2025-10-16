from datasets import load_dataset
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores.base import VectorStoreRetriever
from config.settings import settings

from tqdm import tqdm
import os

from pathlib import Path

class VectorStore:
    def __init__(self):
        self.embedding = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=settings.GEMINI_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


    def load_documents(self):
        self.ds = load_dataset("ShenLab/MentalChat16K")


    def _example_to_document(self) -> list[Document]:
        assert self.ds is not None, "Dataset not loaded"
        docs: list[Document] = []

        for ex in tqdm(self.ds["train"], desc="Building chunks"):
            assistant_output = ex.get("output") or ""

            chunks = self.text_splitter.split_text(assistant_output)

            for chunk in chunks:
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": "MentalChat16K",
                        }
                    )
                )

        return docs


    def build(self) -> FAISS:
        self.load_documents()
        docs = self._example_to_document()

        self.vectorstore = FAISS.from_documents(docs, self.embedding)
        os.makedirs(os.path.dirname(settings.INDEX_DIR), exist_ok=True)
        self.vectorstore.save_local(settings.INDEX_DIR)

        return self.vectorstore
    

    def load(self) -> FAISS:
        assert settings.INDEX_DIR is not None, "INDEX_DIR not set"
        self.vectorstore = FAISS.load_local(
            settings.INDEX_DIR, self.embedding, allow_dangerous_deserialization=True
        )
        return self.vectorstore
    

    def get_retriever(self, k: int = 3) -> VectorStoreRetriever:
        assert self.vectorstore is not None, "Vectorstore not built/loaded"
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


def _count_chunks(chunk_size=1000, chunk_overlap=100):
    ds = load_dataset("ShenLab/MentalChat16K")["train"]

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    total, per_sample = 0, []
    for ex in tqdm(ds, desc="Dry-run chunking"):
        text = f"User: {ex.get('input','')}\nAssistant: {ex.get('output','')}"
        n = len(splitter.split_text(text))
        total += n
        per_sample.append(n)
    print(f"Samples: {len(per_sample)}, Total chunks: {total}, Avg chunks/sample: {sum(per_sample)/len(per_sample):.2f}")


if __name__ == "__main__":

    # _count_chunks()
    vectorstore = VectorStore()
    if not Path(settings.INDEX_DIR).exists():
        print("Index not found, building...")
        vectorstore.build()
    else:
        print("Loading existing FAISS index...")
        vectorstore.load()

    # simple test
    query = "I feel anxious all the time and can't sleep well."
    results = vectorstore.get_retriever(k=3).invoke(query)
    for r in results:
        print("---- Page Content ----")
        print(r.page_content[:400])
        print(r.metadata)