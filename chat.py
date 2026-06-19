import time

import gradio as gr


def model_a_response(message, history):
    time.sleep(1)
    return f"Model A says:\n\n{message}"


def model_b_response(message, history):
    time.sleep(1)
    return f"Model B says:\n\n{message}"


def send_to_both(message, history_a, history_b):
    response_a = model_a_response(message, history_a)
    response_b = model_b_response(message, history_b)

    history_a = history_a + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_a},
    ]

    history_b = history_b + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_b},
    ]

    return "", history_a, history_b


with gr.Blocks() as demo:
    gr.Markdown("# Compare two models")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Model A")
            chat_a = gr.Chatbot()

        with gr.Column():
            gr.Markdown("## Model B")
            chat_b = gr.Chatbot()

    with gr.Row():
        textbox = gr.Textbox(
            placeholder="Ask both models the same question...", scale=4
        )
        submit = gr.Button("Send", scale=1)

    submit.click(
        send_to_both,
        inputs=[textbox, chat_a, chat_b],
        outputs=[textbox, chat_a, chat_b],
    )

    textbox.submit(
        send_to_both,
        inputs=[textbox, chat_a, chat_b],
        outputs=[textbox, chat_a, chat_b],
    )


if __name__ == "__main__":
    demo.launch()
