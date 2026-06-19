import time

import gradio as gr

MODELS = {
    "Model 1": lambda x: f"Model 1 response: {x}",
    "Model 2": lambda x: f"Model 2 response: {x}",
    "Model 3": lambda x: f"Model 3 response: {x}",
}


def run_model(model_name, message):
    time.sleep(1)
    return MODELS[model_name](message)


def send(message, model_a, model_b, history_a, history_b):

    response_a = run_model(model_a, message)
    response_b = run_model(model_b, message)

    history_a = history_a + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_a},
    ]

    history_b = history_b + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_b},
    ]

    return "", history_a, history_b


def build_app():

    with gr.Blocks() as demo:

        gr.Markdown("# Model comparison")

        with gr.Row():

            with gr.Column():
                model_a = gr.Dropdown(
                    choices=list(MODELS.keys()), value="Model 1", label="Model A"
                )

                chat_a = gr.Chatbot(label="Model A")

            with gr.Column():
                model_b = gr.Dropdown(
                    choices=list(MODELS.keys()), value="Model 2", label="Model B"
                )

                chat_b = gr.Chatbot(label="Model B")

        prompt = gr.Textbox(placeholder="Ask both models...")

        send_btn = gr.Button("Send")

        send_btn.click(
            send,
            inputs=[prompt, model_a, model_b, chat_a, chat_b],
            outputs=[prompt, chat_a, chat_b],
        )

        prompt.submit(
            send,
            inputs=[prompt, model_a, model_b, chat_a, chat_b],
            outputs=[prompt, chat_a, chat_b],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
