import os
import glob
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

RESOURCES_DIR = "rag/resources"
VECTORSTORE_PATH = "vectorstore_faiss"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def init_models():
    """initialisation des deux modèles (conversation+plongement)"""
    model = AzureChatOpenAI(
        # TODO: variabliser dans le .env
        azure_endpoint=os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT",
                                      "https://menoua.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview"),
        azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-41"),
        api_version=os.environ.get("AZURE_OPENAI_CHAT_API_VERSION", "2025-01-01-preview"),
        temperature=0,
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        chunk_size=64,
    )

    return model, embeddings

def load_pdf_documents(resources_dir: str) -> list:
    pdf_files = glob.glob(os.path.join(resources_dir, "*.pdf"))

    all_docs = []
    for pdf_path in pdf_files:
        print(f"Chargement de: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  -> {len(docs)} pages chargées")

    print(f"\nTotal: {len(all_docs)} pages chargées depuis {len(pdf_files)} fichiers")
    return all_docs


def split_documents(docs: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(docs)
    print(f"Documents découpés en {len(splits)} morceaux")
    return splits


def create_or_load_vectorstore(embeddings, resources_dir: str = RESOURCES_DIR,
                               vectorstore_path: str = VECTORSTORE_PATH,
                               force_rebuild: bool = False) -> FAISS:

    # vérifie si la bdd existe déjà
    if os.path.exists(vectorstore_path) and not force_rebuild:
        print(f"Chargement de la BDD existante depuis {vectorstore_path}...")
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"BDD chargée avec succès")
        return vectorstore

    print("Création d'une nouvelle BDD...")

    docs = load_pdf_documents(resources_dir)
    splits = split_documents(docs)

    print("Création des plongements et de la BDD FAISS...")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    vectorstore.save_local(vectorstore_path)
    print(f"BDD sauvegardée dans {vectorstore_path}")

    return vectorstore

_model = None
_retriever = None


def create_retriever_tool(vectorstore):
    global _retriever
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    @tool
    def retrieve_documents(query: str) -> str:
        """recherche et retourne des informations pertinentes depuis les documents PDF."""
        docs = _retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    return retrieve_documents

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n"
    "Here is the retrieved document:\n\n{context}\n\n"
    "Here is the user question: {question}\n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:\n"
    "-------\n"
    "{question}\n"
    "-------\n"
    "Formulate an improved question:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks.\n"
    "Use the following pieces of retrieved context to answer the question.\n"
    "If you don't know the answer, just say that you don't know.\n"
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question}\n"
    "Context: {context}"
)

class GradeDocuments(BaseModel):
    """évalue la pertinence des documents"""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def generate_query_or_respond(state: MessagesState):
    """génère une requête de recherche ou répond directement"""
    global _model, _retriever_tool
    response = _model.bind_tools([_retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """évalue la pertinence des documents récupérés"""
    global _model
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = _model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )

    if response.binary_score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


def rewrite_question(state: MessagesState):
    """reformule la question pour améliorer la recherche"""
    global _model
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = _model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}


def generate_answer(state: MessagesState):
    """génère une réponse basée sur le contexte récupéré"""
    global _model
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = _model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


def build_rag_agent(model, retriever_tool):
    """construit le graphe de l'agent RAG avec LangGraph"""
    global _model, _retriever_tool
    _model = model
    _retriever_tool = retriever_tool

    workflow = StateGraph(MessagesState)

    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")

    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    graph = workflow.compile()
    return graph

def chat_with_agent(agent, question: str, verbose: bool = True):
    if verbose:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)

    response_content = ""
    for chunk in agent.stream({"messages": [{"role": "user", "content": question}]}):
        for node, update in chunk.items():
            if verbose:
                print(f"\n[Noeud: {node}]")
                update["messages"][-1].pretty_print()

            if node == "generate_answer" or (node == "generate_query_or_respond" and not update["messages"][-1].tool_calls):
                response_content = update["messages"][-1].content

    return response_content


def interactive_mode(agent):
    print("\n" + "="*60)
    print("Agent RAG - Mode interactif")
    print("Tapez 'quit' ou 'exit' pour quitter")
    print("="*60)

    while True:
        try:
            question = input("\nVotre question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("Au revoir!")
                break

            if not question:
                continue

            chat_with_agent(agent, question)

        except KeyboardInterrupt:
            print("\nInterruption. Au revoir!")
            break

def main():
    print("="*60)
    print("Initialisation de l'agent")
    print("="*60)

    print("\n1. Initialisation des modèles Azure OpenAI...")
    model, embeddings = init_models()
    print("Modèles initialisés")

    print("\n2. Préparation de la BDD...")
    vectorstore = create_or_load_vectorstore(
        embeddings,
        resources_dir=RESOURCES_DIR,
        force_rebuild=False
    )

    print("\n3. Création de l'outil de recherche...")
    retriever_tool = create_retriever_tool(vectorstore)
    print("Outil de recherche créé")

    print("\n4. Construction de l'agent...")
    agent = build_rag_agent(model, retriever_tool)
    print("Agent prêt")

    interactive_mode(agent)


if __name__ == "__main__":
    main()