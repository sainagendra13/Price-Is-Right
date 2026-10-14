import gradio as gr

from workflow import app as graph


def hunt():
    # Execute the whole multi-agent workflow
    result = graph.invoke(
        {"results": []},
        config={"recursion_limit": 100}
    )

    rows = [
        [
            d["title"],
            d["offer"],
            d["dnn"],
            d["rag"],
            d["estimate"],
            d["savings"],
            d["url"]
        ]
        for d in result.get("results", [])
    ]

    return rows, result["message"]


with gr.Blocks(title="The Price Is Right") as demo:

    gr.Markdown(
        "# The Price Is Right — Multi-Agent Deal Hunter"
    )

    gr.Markdown(
        "Scan deals → Preprocess → DNN + RAG "
        "→ Ensemble → Plan → Alert"
    )

    button = gr.Button(
        "Find Deals",
        variant="primary"
    )

    table = gr.Dataframe(
        headers=[
            "Product",
            "Offer ($)",
            "DNN ($)",
            "RAG ($)",
            "Estimate ($)",
            "Savings ($)",
            "URL"
        ],
        datatype=[
            "str", "number", "number", "number",
            "number", "number", "str"
        ],
        interactive=False
    )

    message = gr.Textbox(
        label="Agent decision",
        lines=5
    )

    button.click(
        hunt,
        outputs=[table, message]
    )


if __name__ == "__main__":
    demo.launch()