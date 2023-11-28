import json  
import hashlib
import pandas as pd
import io  
import os  
import base64
import glob
from PIL import Image  
from tqdm import tqdm
import random

def process_files(csv_file, video_root):
    df = pd.read_csv(csv_file)
    instruction = [
        "Describe the video concisely.",
        "Provide a brief description of the given video.",
        "Offer a succinct explanation of the video presented.",
        "Summarize the visual content of the video.",
        "Give a short and clear explanation of the subsequent video.",
        "Share a concise interpretation of the video provided.",
        "Present a compact description of the video's key features.",
        "Relay a brief, clear account of the video shown.",
        "Render a clear and concise summary of the video.",
        "Write a terse but informative summary of the video.",
        "Create a compact narrative representing the video presented."
    ]
    json_data = []
    for index, row in tqdm(df.iterrows()):
        videoid = row['videoid']
        caption = row['name']
        page_dir = row['page_dir']
        if pd.isna(videoid) or pd.isna(page_dir):
            continue

        video_file = f'{videoid}.mp4'
        try:
            video_dir = os.path.join(video_root, page_dir, video_file)
        except:
            import pdb; pdb.set_trace()
        if not os.path.exists(video_dir):
            continue

        instruct = random.choice(instruction)
        conv = [
            {'from': 'human', 'value': f'{instruct}\n<image>'},
            {'from': 'gpt', 'value': caption}
        ]
        data_dict = {}
        data_dict['id'] = videoid
        data_dict['video'] = os.path.join(page_dir, video_file)
        data_dict['conversations'] = conv
        json_data.append(data_dict)

    print('Total', len(json_data), 'videos')
    return json_data  

def write_json(json_data, output_file):
    json.dump(json_data, open(output_file, 'w'), indent=2)
  
if __name__ == '__main__':  
    csv_root = './data/WebVid/results_2M_train_50'
    video_root = './data//LLaMA-VID-Pretrain/videos'
    llava_file = './data/LLaMA-VID-Pretrain/blip_laion_cc_sbu_558k.json' 
    output_file = './data/WebVid/llava_webvid.json'
    llava_data = json.load(open(llava_file, 'r'))

    for i in tqdm(range(5)):
        csv_file = os.path.join(csv_root, f'{i}.csv')
        webvid_data = process_files(csv_file, video_root)
        llava_data.extend(webvid_data)
        print(len(llava_data))

    write_json(llava_data, output_file)
