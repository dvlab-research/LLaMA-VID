import os
import glob
import math
import json
import torch
import pickle
import argparse
import numpy as np
from PIL import Image, TarIO
from tqdm import tqdm
from llamavid.model.multimodal_encoder.eva_vit import EVAVisionTowerLavis
import tarfile
import io
import re
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--video_dir", required=True, help="Path to read the videos from.")
    parser.add_argument("--files_dir", required=True, help="Path to read the shot detection result from.")
    parser.add_argument("--feat_dir", required=True, help="The output dir to save the features in.")
    parser.add_argument("--vision_tower", default="./model_zoo/LAVIS/eva_vit_g.pth", help="Vision backbone to process the video.")
    parser.add_argument("--image_processor", defalut="./llamavid/processor/clip-patch14-224", help="Image processor to pre-process the video.")
    parser.add_argument("--index", type=int, default=0, help="index of chunk.")
    parser.add_argument("--chunk", type=int, default=1, help="number of chunk.")
    parser.add_argument("--infer_batch", required=False, type=int, default=48,
                        help="Number of frames/images to perform batch inference.")
    args = parser.parse_args()
    return args


def get_second(time_str):
    time_obj = datetime.datetime.strptime(time_str, '%H:%M:%S,%f')
    seconds = (time_obj - datetime.datetime(1900, 1, 1)).total_seconds()
    return seconds


def load_subtitles(file_path):

    def check(subtitle):
        if subtitle.get('text', None) is None: return False 
        if subtitle.get('start_time', None) is None: return False 
        if subtitle.get('end_time', None) is None: return False 
        return True

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()

    subtitles = []
    subtitle = {}
    for line in lines:
        line = line.replace('\x00', '').strip()
        line = line.replace('...', '').strip()
        if not line: continue
        if line.isdigit():
            if check(subtitle):
                subtitles.append(subtitle)
                subtitle = {}
        elif ' --> ' in line:
            start, end = line.split(' --> ')
            subtitle['start_time'] = get_second(start)
            subtitle['end_time'] = get_second(end)
        else:
            if subtitle.get('text', None):
                subtitle['text'] += ' ' + line
            else:
                subtitle['text'] = line
    
    if check(subtitle):
        subtitles.append(subtitle)
        subtitle = {}
    
    return subtitles


def load_video(video_path):
    video_file = tarfile.open(video_path, 'r')
    image_file = [x for x in video_file.getmembers() if '.jpg' in x.name]
    image_data = [video_file.extractfile(x).read() for x in image_file]
    image_data = [Image.open(io.BytesIO(x)) for x in image_data]

    file_name = [x.name for x in image_file]
    indexed_list = list(enumerate(file_name))
    indexed_list = sorted(indexed_list, key=lambda x: x[1])
    image_data = [image_data[x] for x, _ in indexed_list]
    return image_data


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
    video_dir = args.video_dir
    feat_dir = args.feat_dir
    files_dir = args.files_dir
    infer_batch = args.infer_batch
    os.makedirs(feat_dir, exist_ok=True)

    # Initialize the CLIP model
    vision_tower = EVAVisionTowerLavis(args.vision_tower, args.image_processor, args=None).cuda()
    vision_tower.eval()
    image_processor = vision_tower.image_processor


    video_files = glob.glob(video_dir + "/*.tar")
    video_files.sort()
    video_files_select = []
    for video_path in video_files:
        imdb_id = os.path.basename(video_path).split('.')[0]
        srt_path = os.path.join(files_dir, 'subtitle', f'{imdb_id}.srt')
        if os.path.exists(srt_path):
            video_files_select.append(video_path)
    video_files = video_files_select


    block = (len(video_files) + args.chunk - 1) // args.chunk
    start = args.index * block
    end = min((args.index + 1) * block, len(video_files))
    print('Process', start, 'to', end)
    video_files = video_files[start:end]


    for video_path in tqdm(video_files):

        imdb_id = os.path.basename(video_path).split('.')[0]
        feat_path = os.path.join(feat_dir, f'{imdb_id}.pkl')
        meta_path = os.path.join(files_dir, 'meta', f'{imdb_id}.json')
        shot_path = os.path.join(files_dir, 'shot', f'{imdb_id}.txt')
        subtitle_path = os.path.join(files_dir, 'subtitle', f'{imdb_id}.srt')
        script_path = os.path.join(files_dir, 'script', f'{imdb_id}.script')

        if not os.path.exists(subtitle_path): continue
        if not os.path.exists(script_path): continue
        if args.compute_feat and os.path.exists(feat_path): continue

        with open(shot_path, 'r') as file:
            lines = file.readlines()
            frame_idx = [int(num) for line in lines for num in line.split()[2:]]
            num_frames = int(lines[-1].split()[1])
        subtitles = load_subtitles(subtitle_path)
        meta_info = json.load(open(meta_path, 'r'))

        # duration = math.ceil(subtitles[-1]['end_time'])
        duration = meta_info['version'][0]['runtime']
        duration = int(re.search(r'\d+', duration).group()) * 60
        fps = num_frames / duration
        frame_time = [x / fps for x in frame_idx]
        video_input = load_video_input(subtitles, frame_time)

        image_data = load_video(video_path)
        video_tensor = image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half()

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
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()