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

def write_json(json_data, output_file):
    json.dump(json_data, open(output_file, 'w'), indent=2)
  
if __name__ == '__main__':
    video_file = 'data/video-chatgpt/video_chatgpt_training_all.json'
    llava_file = 'data/LLaMA-VID-Finetune/llava_v1_5_mix665k.json' 
    output_file = 'data/llava_video_instruct.json'
    llava_data = json.load(open(llava_file, 'r'))
    video_data = json.load(open(video_file, 'r'))
    print('LLaVA has', len(llava_data), 'pairs')

    for item in tqdm(video_data):
        item['video'] = item['video'].replace('pkl', 'mp4')
        item['conversations'][0]['value'] = item['conversations'][0]['value'].replace('<video>', '<image>')
        llava_data.append(item)

    print('LLaVA has', len(llava_data), 'pairs now')
    write_json(llava_data, output_file)
