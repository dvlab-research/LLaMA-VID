import os
import math
import json
import torch
import pickle
import argparse
import numpy as np
from decord import VideoReader, cpu
from transformers import CLIPVisionModel, CLIPImageProcessor
from llamavid.model.multimodal_encoder.eva_vit import EVAVisionTowerLavis
import whisper


def parse_args():
    parser = argparse.ArgumentParser(description="Extract CLIP feature and subtitles for a video")

    parser.add_argument("--video_file", required=True, help="Path to read the videos from.")
    parser.add_argument("--feat_dir", required=True, help="The output dir to save the features in.")
    parser.add_argument("--infer_batch", required=False, type=int, default=48,
                        help="Number of frames/images to perform batch inference.")
    parser.add_argument("--fps", required=False, type=int, default=1, help="video fps")
    args = parser.parse_args()
    return args


def load_subtitles(video_file):
    MIN_CONFIDENCE = -2
    model = whisper.load_model("large") 
    result = model.transcribe(video_file)
    timestamps = result["segments"]
    timestamps = [segment for segment in timestamps if segment['avg_logprob'] >= MIN_CONFIDENCE]
    subtitles = []
    for content in timestamps:
        text = content['text']
        if text[0] == ' ': text = text[1:]
        subtitle = {}
        subtitle['start_time'] = content['start']
        subtitle['end_time'] = content['end']
        subtitle['text'] = text
        subtitles.append(subtitle)
    return subtitles


def load_video(video_path, sample_fps=1):
    print(video_path)
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    video_fps = vr.get_avg_fps()
    sample_gap = round(video_fps / sample_fps)
    sample_frame_idx = [i for i in range(0, total_frame_num, sample_gap)]
    img_array = vr.get_batch(sample_frame_idx).asnumpy()
    frame_time = [x / video_fps for x in sample_frame_idx]
    return img_array, frame_time


def load_video_input(subtitles, frame_time):
    point = 0 
    text_list = [[] for i in range(len(frame_time))]
    for item in subtitles:
        end_time = item['end_time']
        text = item['text']
        while point < len(frame_time) and frame_time[point] < end_time:
            point += 1
        text_list[point - 1].append(text)
    video_input = ''
    for text in text_list:
        video_input += '<image>' + ' '.join(text)
    return video_input


def main():
    args = parse_args()
    video_file = args.video_file
    feat_dir = args.feat_dir
    infer_batch = args.infer_batch
    feat_path = os.path.join(feat_dir, os.path.basename(video_file)).replace('.mp4', '.pkl').replace('.mkv', '.pkl')
    print(feat_path)

    # Initialize the CLIP model
    vision_tower = "./model_zoo/LAVIS/eva_vit_g.pth"
    image_processor = "./model_zoo/OpenAI/clip-vit-large-patch14"
    vision_tower = EVAVisionTowerLavis(vision_tower, image_processor, args=None).cuda()
    vision_tower.eval()
    image_processor = vision_tower.image_processor

    subtitles = load_subtitles(video_file)
    video, frame_time = load_video(video_file, sample_fps=args.fps)
    video_input = load_video_input(subtitles, frame_time)

    video_tensor = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half()
    print(video_tensor.shape)

    n_chunk = len(video_tensor)
    video_features = torch.FloatTensor(n_chunk, 257, 1408).fill_(0)
    n_iter = int(math.ceil(n_chunk / float(infer_batch)))
    for i in range(n_iter):
        min_ind = i * infer_batch
        max_ind = min((i + 1) * infer_batch, n_chunk)
        video_batch = video_tensor[min_ind:max_ind].cuda()
        batch_features = vision_tower(video_batch)
        video_features[min_ind:max_ind] = batch_features.detach().cpu()
    video_features = video_features.numpy().astype("float16")
    print(video_features.shape)
    video_info = dict(feats=video_features, inputs=video_input)

    with open(feat_path, 'wb') as f:
        pickle.dump(video_info, f)

if __name__ == "__main__":
    main()
