import os
import csv
import json
import threading
from time import time
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from fastapi.responses import FileResponse
import pathlib 

load_dotenv(dotenv_path="../.env")
DB_FAISS_PATH = pathlib.Path("vectorstore/dbfaiss")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
ALLOWED_MODELS = ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"]
CSV_LOG_PATH = "logs/query_logs.csv"
JSON_BATCH_OUTPUT = "logs/batch_answers.json"
PERFORMANCE_CSV_PATH = "logs/performance_test.csv"
MEDQUAD_PATH = "../data/medquad.csv"

app = FastAPI(title="Medical RAG Chatbot")

lock = threading.Lock()
query_data_store = {}

class QueryRequest(BaseModel):
    model_name: str
    messages: List[str]


class ManualEvalRequest(BaseModel):
    query: str
    relevance: int
    accuracy: int
    fluency: int
    sources_flag: bool


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL)
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)


def get_llm_response(model_name, question, context_docs):
    llm = ChatGroq(
        model=model_name,
        groq_api_key=GROQ_API_KEY,
        temperature=0.0,
    )
    context = "\n".join([doc.page_content for doc in context_docs])
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
    messages = [
        SystemMessage(content="You are a helpful medical assistant."),
        HumanMessage(content=prompt)
    ]
    return llm.invoke(messages).content


def estimate_query_length(text: str) -> str:
    word_count = len(text.split())
    if word_count < 5:
        return "Kisa"
    elif word_count < 10:
        return "Orta"
    else:
        return "Uzun"


def log_combined_to_csv(data: ManualEvalRequest):
    os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(CSV_LOG_PATH)

    with open(CSV_LOG_PATH, mode="a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Query", "Query_Length", "Retrieval_Time_MS", "Generation_Time_MS",
            "Total_Time_MS", "Answer", "Reference",
            "Relevance", "Accuracy", "Fluency", "Sources_Flag"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        with lock:
            stored = query_data_store.get(data.query)

        if not stored:
            stored = {
                "query_len_label": "",
                "retrieval_time": "",
                "generation_time": "",
                "total_time": "",
                "answer": "",
                "sources": []
            }

        writer.writerow({
            "Query": data.query,
            "Query_Length": stored["query_len_label"],
            "Retrieval_Time_MS": stored["retrieval_time"],
            "Generation_Time_MS": stored["generation_time"],
            "Total_Time_MS": stored["total_time"],
            "Answer": stored["answer"],
            "Reference": " | ".join(stored["sources"]),
            "Relevance": data.relevance,
            "Accuracy": data.accuracy,
            "Fluency": data.fluency,
            "Sources_Flag": "Yes" if data.sources_flag else "No"
        })


@app.post("/query")
def query_endpoint(request: QueryRequest):
    if request.model_name not in ALLOWED_MODELS:
        return {"error": f"Invalid model name. Choose from: {ALLOWED_MODELS}"}

    try:
        query = request.messages[-1]
        query_len_label = estimate_query_length(query)

        start_retrieval = time()
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        docs = retriever.get_relevant_documents(query)
        retrieval_time = int((time() - start_retrieval) * 1000)

        start_generation = time()
        answer = get_llm_response(request.model_name, query, docs)
        generation_time = int((time() - start_generation) * 1000)

        total_time = retrieval_time + generation_time

        formatted_sources = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            formatted_sources.append(f"{source} - page {page}")

        with lock:
            query_data_store[query] = {
                "query_len_label": query_len_label,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "answer": answer,
                "sources": formatted_sources,
            }

        return {
            "question": query,
            "answer": answer,
            "sources": formatted_sources,
            "latency": f"{total_time} ms"
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/manual_evaluation")
def manual_evaluation(eval_request: ManualEvalRequest):
    try:
        log_combined_to_csv(eval_request)
        return {"status": "Evaluation received."}
    except Exception as e:
        return {"error": str(e)}


@app.get("/batch_generate_answers")
def batch_generate_answers():
    try:
        df = pd.read_csv(MEDQUAD_PATH)

        if "question" not in df.columns or "answer" not in df.columns:
            return {"error": "Missing 'question' or 'answer' column in CSV."}

        sample_df = df.sample(n=100, random_state=42)
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        results = []

        for _, row in sample_df.iterrows():
            question = row["question"]
            ground_truth = row["answer"]

            try:
                docs = retriever.get_relevant_documents(question)
                model_answer = get_llm_response(ALLOWED_MODELS[0], question, docs)

                results.append({
                    "question": question,
                    "model_answer": model_answer,
                    "ground_truth": ground_truth
                })
            except Exception as e:
                results.append({
                    "question": question,
                    "model_answer": f"[Error: {str(e)}]",
                    "ground_truth": ground_truth
                })

        os.makedirs(os.path.dirname(JSON_BATCH_OUTPUT), exist_ok=True)
        with open(JSON_BATCH_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return FileResponse(JSON_BATCH_OUTPUT, media_type="application/json", filename="batch_answers.json")

    except Exception as e:
        return {"error": str(e)}


@app.get("/performance_test")
def performance_test_generate():
    try:
        df = pd.read_csv(MEDQUAD_PATH)

        if "question" not in df.columns:
            return {"error": "'question' column missing."}

        sample_df = df.sample(n=100, random_state=42).reset_index(drop=True)
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

        os.makedirs(os.path.dirname(PERFORMANCE_CSV_PATH), exist_ok=True)
        with open(PERFORMANCE_CSV_PATH, mode="w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "Query", "Query_Length", "Retrieval_Time_MS",
                "Generation_Time_MS", "Total_Time_MS", "Answer"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for _, row in sample_df.iterrows():
                question = row["question"]
                query_len_label = estimate_query_length(question)

                try:
                    start_retrieval = time()
                    docs = retriever.get_relevant_documents(question)
                    retrieval_time = int((time() - start_retrieval) * 1000)

                    start_generation = time()
                    answer = get_llm_response(ALLOWED_MODELS[0], question, docs)
                    generation_time = int((time() - start_generation) * 1000)

                    total_time = retrieval_time + generation_time

                    writer.writerow({
                        "Query": question,
                        "Query_Length": query_len_label,
                        "Retrieval_Time_MS": retrieval_time,
                        "Generation_Time_MS": generation_time,
                        "Total_Time_MS": total_time,
                        "Answer": answer
                    })

                except Exception as e:
                    writer.writerow({
                        "Query": question,
                        "Query_Length": query_len_label,
                        "Retrieval_Time_MS": -1,
                        "Generation_Time_MS": -1,
                        "Total_Time_MS": -1,
                        "Answer": f"[Error: {str(e)}]"
                    })

        return FileResponse(PERFORMANCE_CSV_PATH, media_type="text/csv", filename="performance_test.csv")

    except Exception as e:
        return {"error": str(e)}
