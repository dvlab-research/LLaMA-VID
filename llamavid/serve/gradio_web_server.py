import argparse
import datetime
import json
import os
import time

import gradio as gr
import requests

import tempfile
import shutil

from llamavid.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llamavid.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import hashlib


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "LLaVA Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(
                value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown.update(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5


def add_text(state, text, image, video, image_process_mode, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and (image is None and video is None):
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None, None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None, None) + (
                no_change_btn,) * 5
    if image is not None:
        text = (text, image, None, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()
    if video is not None:
        text = (text, None, video, image_process_mode)
        if len(state.get_videos()) > 0:
            state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5


def http_bot(state, model_selector, movie_part, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        template_name = "vicuna_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    all_videos = state.get_videos()
    '''
    for video in all_videos:
        t = datetime.datetime.now()
        filename = next(tempfile._get_candidate_names())
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{filename}.mp4")
        shutil.copyfile(video, filename)
    '''

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "movie_part": int(movie_part),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
        "videos": f'List of {len(state.get_videos())} videos: {all_videos}',
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()
    pload['videos'] = state.get_videos()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=30)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "videos": all_videos,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

title_markdown = ("""
# LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models
[[Project Page]](https://llama-vid.github.io/) [[Paper]](https://arxiv.org/abs/2311.17043) [[Code]](https://github.com/dvlab-research/LLaMA-VID) [[Model]](https://huggingface.co/YanweiLi)
""")

introduce_markdown = ("""
### Usage
Due to limit computing resources for demo, we currently do not support chat with customized long videos.
If you want to chat with customized short video, please select **llama-vid-vicuna-7b-short**.
If you want to chat with preloaded long movie, please select **llama-vid-vicuna-7b-long**.
Since we deploy LLaMA-VID on a single 3090 GPU, a movie is devided to **5 or 6 parts** to save memory. If you want to inference on a whole 3-hour movie, please consider a larger memory.
Please **select a movie in the following collection**, and also **set the movie part in the above box**. You can chat with the movie now! Please rechoose the movie for part change.
""")

plot_markdown = ("""
### Plot introduction
Here we introduct the polt of each part for your convenience.

**Titanic**
Part 0 (0->30 min): *Treasure hunter Brock Lovett searches for a valuable diamond aboard the Titanic wreck, leading to the revelation of the ship's history and a love story through the memories of survivor Rose Dawson Calvert.*
Part 1 (30->60 min): *Jack saves Rose from jumping off the ship and they are invited to dine with first-class passengers.*
Part 2 (60->90 min): *Jack and Rose develop a forbidden friendship onboard the Titanic, as Rose begins to question her impending marriage to Cal and discovers her growing feelings for Jack.*
Part 3 (90->120 min): *Rose and Jack witness the ship’s collision with an iceberg and overhear discussions of its impending sinking. Cal, Rose’s fiance, frames Jack for theft, but as the ship sinks, Rose rescues Jack and they make a desperate attempt to survive together.*
Part 4 (120->150 min): *In Titanic, Jack and Rose share a romantic moment, but are chased by Cal. They narrowly escape drowning and Cal is left behind, unable to escape the sinking ship.*
Part 5 (150->180 min): *The Titanic sink, Jack sacrifices himself for Rose's survival, and she is later rescued and takes on a new identity.*

**Avatar**
Part 0 (0->30 min): *A crippled war veteran Jake Sully takes over his deceased twin brother's contract to travel to Pandora, a moon with a lush rainforest and indigenous population called the Na'vi, in order to study and mine a valuable mineral.*
Part 1 (30->60 min): *Jake and his team explore Pandora, encountering dangerous creatures, and Jake eventually becomes a part of the Na'vi tribe after being saved by Neytiri.*
Part 2 (60->90 min): *Grace moves her operation to the Hallelujah Mountains to avoid the RDA officials and military, and Jake learns the Na'vi ways, bonds with a banshee, and gains the respect of the Na'vi warriors.*
Part 3 (90->120 min): *Jake tries to negotiate their relocation, but his efforts are thwarted by the Colonel’s destructive actions, leading to a battle that results in the destruction of Hometree and the death of Neytiri’s father, Eytukan.*
Part 4 (120->150 min): *Jake, after escaping arrest, tames a legendary creature to regain the Na’vi’s trust, and rallies them to fight against humans who plan to destroy their sacred Tree of Souls.*
Part 5 (150->180 min): *Jake defeats Quaritch in a duel, Neytiri saves him, and he permanently becomes his avatar on Pandora.*

**The Avengers**
Part 0 (0->30 min): *S.H.I.E.L.D. agent Nick Fury activates the Avengers Initiative after Loki steals the Tesseract, an energy source of unknown potential, and plans to use it to conquer Earth.*
Part 1 (30->60 min): *The Avengers team starts to assemble as Black Widow recruits Bruce Banner, Iron Man is brought in on the mission, and Captain America is tasked to capture Loki, culminating in a successful capture after a brief confrontation.*
Part 2 (60->90 min): *With Loki in custody aboard the S.H.I.E.L.D. Helicarrier, tensions rise among the Avengers due to their clashing personalities and Loki's manipulations, leading to internal conflict and the unleashing of the Hulk.*
Part 3 (90->120 min): *The team suffers setbacks as Hawkeye leads an attack on the Helicarrier, Thor and Hulk fight, and Loki escapes, killing Agent Coulson, which ultimately becomes the catalyst for the Avengers to unite.*
Part 4 (120->150 min): *The Avengers come together to defend New York City against Loki's alien army in a climactic battle, ultimately capturing Loki and restoring peace while proving themselves a capable team.*

**Intersteller**
Part 0 (0->30 min): *In a deteriorating agrarian society plagued by crop blight, former NASA pilot Cooper and his daughter Murphy discover coordinates to a secret NASA facility, where they learn that humanity faces extinction as crops fail annually due to severe dust storms and climate change.*
Part 1 (30->60 min): *Regardless of the objections of his daughter Murphy, Cooper piloted the spacecraft Endurance to perform a mission in space to try to find a new home for human.*
Part 2 (60->90 min): *Cooper and his team entered Miller's planet but found it uninhabitable, causing them to lose 23 years. On earth, Murphy grow up and worked with Dr. Brand to solve the equations of gravity.*
Part 3 (90->120 min): *Cooper and his team selects Mann's planet based on recent data, but it turns out to be hostile and Mann's true intentions are revealed, leading to a series of life-threatening events.*
Part 4 (120->150 min): *Cooper and Amelia slingshot around Gargantua, sacrificing themselves to collect data on the singularity, and end up in an extra-dimensional 'tesseract' where Cooper communicates with Murphy to save humanity.*
Part 5 (150->170 min): *Cooper successfully transmitted the data to Murphy, who solved the gravity equation and saved mankind. Older Murphy reunites with Cooper.*

**Forrest Gump**
Part 0 (0->30 min): *Forrest Gump, a man with a low IQ from Alabama, excels in football due to his exceptional speed, earning a scholarship and the chance to meet President Kennedy following his team's championship victory.*
Part 1 (30->60 min): *Forrest Gump enlists in the Army, forms friendships with Bubba and Lieutenant Dan in Vietnam, saves fellow soldiers in battle where Bubba dies and Lieutenant Dan is injured, and awarded the Medal of Honor for Vietnam War heroism.*
Part 2 (60->90 min): *Forrest Gump becomes a ping-pong player and become celebrity, he playing against Chinese teams on a goodwill tour. Then he start a successful shrimping business with Lieutenant Dan, to fulfilling his promise to Bubba.*
Part 3 (90->120 min): *After mother's death and a rejected marriage proposal to Jenny, Forrest Gump embarks on a famous cross-country run for three and a half years.*
Part 4 (120->150 min): *Forrest reunites with Jenny and learns he has a son; they move to Alabama and marry before Jenny's death, leaving Forrest to care for their son as the film closes with a symbolic white feather drifting skyward.*
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""

def build_demo(embed_mode):
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="LLaMA-VID", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                imagebox = gr.Image(label="Movie Poster", type="pil", interactive=False)
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                with gr.Accordion("Movie Part", open=True) as parameter_row:
                    movie_part = gr.Radio(choices=[0, 1, 2, 3, 4, 5], value=0, interactive=True, label="Part selection (plot introduced below)")

                videobox = gr.Video(label="Input Video")

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    # movie_part = gr.Slider(minimum=0, maximum=5, value=1, step=1, interactive=True, label="Movie Part",)
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

            with gr.Column(scale=6):
                chatbot = gr.Chatbot(elem_id="chatbot", label="LLaMA-VID Chatbot", height=650)
                with gr.Row():
                    with gr.Column(scale=6):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)
        
        description_titanic = "Titanic (Choose Part 0-5 in the above movie part)"
        description_avatar = "Avatar (Choose Part 0-5 in the above movie part)"
        description_avengers = "The Avengers (Choose Part 0-4 in the above movie part)"
        description_intersteller = "Intersteller (Choose Part 0-5 in the above movie part)"
        description_gump = "Forrest Gump (Choose Part 0-4 in the above movie part)"

        if not embed_mode:
            gr.Markdown(introduce_markdown)

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        gr.Examples(examples=[
            [f"{cur_dir}/examples/Titanic.jpg", description_titanic, "Write a brief summary of this movie part."],
            [f"{cur_dir}/examples/Avatar.png", description_avatar, "How does human work on Pandora?"],
            [f"{cur_dir}/examples/Avengers.jpg", description_avengers, "Who is the main antagonist in this movie, and what is his goal?"],
            [f"{cur_dir}/examples/Interstellar.jpg", description_intersteller, "What happens in this video?"],
            [f"{cur_dir}/examples/Forrest_Gump.jpg", description_gump, "What did Forrest Gump do in this movie part?"],
        ], inputs=[imagebox, textbox, textbox])

        if not embed_mode:
            gr.Markdown(plot_markdown)
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(upvote_last_response,
            [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
        downvote_btn.click(downvote_last_response,
            [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
        flag_btn.click(flag_last_response,
            [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
        regenerate_btn.click(regenerate, [state, image_process_mode],
            [state, chatbot, textbox, imagebox, videobox] + btn_list).then(
            http_bot, [state, model_selector, movie_part, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list)
        clear_btn.click(clear_history, None, [state, chatbot, textbox, imagebox, videobox] + btn_list)

        textbox.submit(add_text, [state, textbox, imagebox, videobox, image_process_mode], [state, chatbot, textbox, imagebox, videobox] + btn_list
            ).then(http_bot, [state, model_selector, movie_part, temperature, top_p, max_output_tokens],
                   [state, chatbot] + btn_list)
        submit_btn.click(add_text, [state, textbox, imagebox, videobox, image_process_mode], [state, chatbot, textbox, imagebox, videobox] + btn_list
            ).then(http_bot, [state, model_selector, movie_part, temperature, top_p, max_output_tokens],
                   [state, chatbot] + btn_list)

        if args.model_list_mode == "once":
            demo.load(load_demo, [url_params], [state, model_selector],
                _js=get_window_url_params)
        elif args.model_list_mode == "reload":
            demo.load(load_demo_refresh_model_list, None, [state, model_selector])
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed)
    demo.queue(
        concurrency_count=args.concurrency_count,
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
