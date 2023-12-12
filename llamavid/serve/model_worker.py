"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid
import pickle
import os
import numpy as np

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial

from llamavid.constants import WORKER_HEART_BEAT_INTERVAL
from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.model.builder import load_pretrained_model
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from llava.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from transformers import TextIteratorStreamer
from threading import Thread

from llamavid.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name,
                 load_8bit, load_4bit, device, args):
        replace_llama_attn_with_flash_attn(inference=True)
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        self.device = device
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device)
        self.is_multimodal = ('llava' in self.model_name.lower()) or ('vid' in self.model_name.lower())
        
        self.model.eval()
        self.model.get_vision_tower().cpu()
        torch.cuda.empty_cache()

        self.dump_model_to_cpu()

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=20)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }
    
    def load_model_from_cpu(self):
        torch.cuda.empty_cache()
        self.model.get_model().vlm_att_encoder.cuda()
        self.model.get_model().vlm_att_projector.cuda()
        self.model.get_model().vlm_att_key_projector.cuda()
        self.model.get_model().vlm_att_val_projector.cuda()
    
    def dump_model_to_cpu(self):
        self.model.get_model().vlm_att_encoder.cpu()
        self.model.get_model().vlm_att_projector.cpu()
        self.model.get_model().vlm_att_key_projector.cpu()
        self.model.get_model().vlm_att_val_projector.cpu()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate_stream(self, params):
        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        videos = params.get("videos", None)
        num_image_tokens = 0

        if len(videos) > 0 and len(images) == 0:
            yield json.dumps({"text": ori_prompt + "Please switch to \'llama-vid-vicuna-7b-short\' to chat with upload short videos. After switch, you can clear the conversaton then retry.", "error_code": 0}).encode() + b"\0"
            return
        
        self.load_model_from_cpu()
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        images = [load_image_from_base64(image) for image in images]
        image = np.array(images[0])

        movie_part = int(params.get("movie_part", 1))

        if image.shape[0] == 595 and image.shape[1] == 400:
            while True:
                video_file = f"/data/LLaMAVID/movie_feats_part/tt0120338_{movie_part}.pkl"
                print(video_file, os.path.exists(video_file), '+++')
                if os.path.exists(video_file): break
                movie_part -= 1
        elif image.shape[0] == 590 and image.shape[1] == 400:
            while True:
                video_file = f"/data/LLaMAVID/movie_feats_part/tt0499549_{movie_part}.pkl"
                if os.path.exists(video_file): break
                movie_part -= 1
        elif image.shape[0] == 582 and image.shape[1] == 400:
            while True:
                video_file = f"/data/LLaMAVID/movie_feats_part/tt0848228_{movie_part}.pkl"
                if os.path.exists(video_file): break
                movie_part -= 1
        elif image.shape[0] == 592 and image.shape[1] == 400:
            while True:
                video_file = f"/data/LLaMAVID/movie_feats_part/tt0816692_{movie_part}.pkl"
                if os.path.exists(video_file): break
                movie_part -= 1
        elif image.shape[0] == 580 and image.shape[1] == 400:
            while True:
                video_file = f"/data/LLaMAVID/movie_feats_part/tt0109830_{movie_part}.pkl"
                if os.path.exists(video_file): break
                movie_part -= 1
        else:
            print(image.shape)
            self.dump_model_to_cpu()
            yield json.dumps({"text": ori_prompt + "Custom video uploads are currently not supported", "error_code": 0}).encode() + b"\0"
            return

        video_info = pickle.load(open(video_file, 'rb'))
        input_prompt = video_info['inputs']
        video = torch.from_numpy(video_info['feats'][:, 1:]).to(device=model.device, dtype=model.dtype)
        images = [video]
        image_args = {"images": images}

        start_prompt = 'Below is a part of movie. Memorize the content and answer my question after this movie part. The movie part start.\n'
        end_prompt = '\nThe movie part end.\n'
        input_prompt = start_prompt + input_prompt + end_prompt

        cur_prompt = prompt.split('<image>\n')[1].split(' ASSISTANT:')[0]
        prompt = prompt.replace('<image>\n', input_prompt)

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_token', 2048)

        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1])

        print(input_ids.shape, '-----', video.shape)

        if max_new_tokens < 1:
            self.dump_model_to_cpu()
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return
        
        torch.cuda.empty_cache()
        
        model.update_prompt([[cur_prompt]])
        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        ))
        thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

        self.dump_model_to_cpu()

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.load_4bit,
                         args.device,
                         args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
