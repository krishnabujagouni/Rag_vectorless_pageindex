
from typing import TypedDict, List, Annotated
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import operator
import json
from langgraph.graph import StateGraph, START, END
from typing import Annotated, Literal, TypedDict



from dotenv import load_dotenv
load_dotenv()


DEFAULT_MODEL       = "gpt-4o-mini"
MAX_PAGES_PER_NODE  = 10
TOP_K_NODES         = 3

llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2)




class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    query: str

    tree_index: dict

    retrieved_nodes: list[dict]

    context: str

    final_answer: str

    next_agent: Literal["retriever_agent", "generator_agent", "end"]

   
    step: int
    


def supervisor_node(state: AgentState) -> dict:
    """
    Decides the routing:
      - If retrieved_nodes is empty → send to retriever_agent
      - If retrieved_nodes is populated but no final_answer → send to generator_agent
      - If final_answer is set → end
    """
    step = state.get("step", 0) + 1

    if state.get("final_answer"):
        next_agent = "end"
        decision   = "Final answer is ready. Routing to END."
    elif not state.get("retrieved_nodes"):
        next_agent = "retriever_agent"
        decision   = "No nodes retrieved yet. Routing to RetrieverAgent."
    else:
        next_agent = "generator_agent"
        decision   = f"{len(state['retrieved_nodes'])} nodes retrieved. Routing to GeneratorAgent."

    print(f"\n[Supervisor | step={step}] {decision}")

    return {
        "next_agent": next_agent,
        "step": step,
        "messages": [AIMessage(content=f"[Supervisor] {decision}", name="supervisor")],
    }



def retriever_agent_node(state: AgentState) -> dict:
    """
    Retriever Agent: reasons over the PageIndex tree structure to find
    the TOP_K_NODES most relevant nodes. No vectors, pure LLM reasoning.
    """
    _RETRIEVER_SYSTEM = """You are a PageIndex Retriever Agent.

Your job is to analyse a document's tree index (titles + summaries only — no raw text)
and use step-by-step reasoning to select the nodes most likely to answer the user's question.

The document tree is provided as JSON. Reason like a human expert scanning a table of contents.
Return ONLY a valid JSON object:
{
  "thinking": "<your step-by-step reasoning>",
  "node_list": ["0001", "0003", ...]
}"""
    

    query      = state["query"]
    tree_index = state["tree_index"]
    nodes      = tree_index["nodes"]
    doc_desc   = tree_index.get("doc_description", "")

    # Build compact tree representation (no raw page text)
    tree_repr = json.dumps(
        [{"node_id": n["node_id"],
          "title":   n["title"],
          "summary": n["summary"],
          "pages":   f"{n['start_page']}–{n['end_page']}"}
         for n in nodes],
        indent=2,
    )

    user_msg = (
        f"Document overview: {doc_desc}\n\n"
        f"Tree Index:\n{tree_repr}\n\n"
        f"Question: {query}\n\n"
        f"Select the {TOP_K_NODES} most relevant nodes."
    )

    response = llm.invoke([
        SystemMessage(content=_RETRIEVER_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    try:
        result   = json.loads(response.content)
        thinking = result.get("thinking", "")
        node_ids = result.get("node_list", [])
    except (json.JSONDecodeError, KeyError):
        node_ids = [n["node_id"] for n in nodes[:TOP_K_NODES]]
        thinking = "JSON parse failed — falling back to first nodes."

    print(f"\n[RetrieverAgent] Reasoning:\n{thinking}")
    print(f"[RetrieverAgent] Selected nodes: {node_ids}")

    node_map       = {n["node_id"]: n for n in nodes}
    retrieved      = [node_map[nid] for nid in node_ids if nid in node_map]

    # Build context string for the GeneratorAgent
    context_parts = []
    for node in retrieved:
        pages_text = "\n".join(
            f"  [Page {p['page_number']}] {p['text']}" for p in node["pages"]
        )
        context_parts.append(
            f"### {node['title']} (pages {node['start_page']}–{node['end_page']})\n{pages_text}"
        )
    context = "\n\n".join(context_parts)

    retriever_msg = (
        f"[RetrieverAgent] Retrieved {len(retrieved)} nodes: "
        f"{[n['title'] for n in retrieved]}"
    )

    return {
        "retrieved_nodes": retrieved,
        "context": context,
        "messages": [AIMessage(content=retriever_msg, name="retriever_agent")],
    }



def generator_agent_node(state: AgentState) -> dict:
    """
    Generator Agent: takes the retrieved context and produces a final answer.
    """

    _GENERATOR_SYSTEM = """You are a PageIndex Generator Agent.

Your job is to synthesise a clear, accurate, and well-cited answer to the user's question
using ONLY the document context provided by the Retriever Agent.

Rules:
- Ground every claim in the provided context.
- Cite page numbers where relevant using the format (p. N).
- If the context does not contain enough information, say so explicitly.
- Be concise but thorough.
- Structure the answer clearly: direct answer first, then supporting details.
"""


    query   = state["query"]
    context = state["context"]

    user_msg = f"Context from document:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = llm.invoke([
        SystemMessage(content=_GENERATOR_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    answer = response.content.strip()
    print(f"\n[GeneratorAgent] Answer generated ({len(answer)} chars)")

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer, name="generator_agent")],
    }

def route_supervisor(state: AgentState) -> str:
    """Return the next node name based on supervisor's decision."""
    return state["next_agent"]

def build_graph():
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor",       supervisor_node)
    graph.add_node("retriever_agent",  retriever_agent_node)
    graph.add_node("generator_agent",  generator_agent_node)

    # Entry point
    graph.add_edge(START, "supervisor")

    # Supervisor conditionally routes
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "retriever_agent": "retriever_agent",
            "generator_agent": "generator_agent",
            "end":             END,
        },
    )

    # After each agent, always go back to supervisor for re-routing
    graph.add_edge("retriever_agent", "supervisor")
    graph.add_edge("generator_agent", "supervisor")

    return graph.compile()