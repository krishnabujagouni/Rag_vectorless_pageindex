from agent import AgentState
from pageindex import load_tree_index, build_tree_index, TREE_INDEX_PATH
from agent import build_graph
from langchain_core.messages import HumanMessage
from pathlib import Path


from dotenv import load_dotenv
load_dotenv()


PDF_PATH = "C:\\Users\\krishna\\pagaindex\\2602.06039v1.pdf"


def main():
    
    
    if not Path(TREE_INDEX_PATH).exists():
        print("📄 No index found. Building tree index from PDF...")
        build_tree_index(PDF_PATH)
        print("✅ Index built!\n")
    else:
        print("✅ Index already exists. Skipping indexing.\n")


    tree_index = load_tree_index(TREE_INDEX_PATH)
    
    rag = build_graph()

    while True:
        question = input("\n❓ Ask a question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        initial_state: AgentState = {
            "messages":        [HumanMessage(content=question)],
            "query":           question,
            "tree_index":      tree_index,
            "retrieved_nodes": [],
            "context":         "",
            "final_answer":    "",
            "next_agent":      "retriever_agent",
            "step":            0,
        }
        
        print(f"\n{'='*60}")
        print(f"QUERY: {question}")
        print('='*60)

        final_state = rag.invoke(initial_state)
        print(f"\n✅ Final Answer:\n{final_state['final_answer']}")


if __name__ == "__main__":
    main()