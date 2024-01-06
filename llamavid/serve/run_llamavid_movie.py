import argparse
import torch
import pickle

from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llamavid.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default="./cache")
    parser.add_argument("--video-file", type=str, required=True)
    parser.add_argument("--video-token", type=int, default=2)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default='vicuna_v1')
    parser.add_argument("--model-max-length", type=int, default=None)
    parser.add_argument("--pure-text", action='store_true', help='use image or not')
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")

    return parser.parse_args()


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """

    replace_llama_attn_with_flash_attn(inference=True)

    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)
    
    video_info = pickle.load(open(args.video_file, 'rb'))
    input_prompt = video_info['inputs']
    if args.pure_text:
        print('Pure text')
        input_prompt = input_prompt.replace('<image>', '')
        video = None
    else:
        print('Text with video')
        # replace the default image token with multiple tokens
        input_prompt = input_prompt.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN * args.video_token)
        video = torch.from_numpy(video_info['feats'][:, 1:]).cuda().half()
        video = [video]

    start_prompt = 'Below is a movie. Memorize the content and answer my question after watching this movie.'
    end_prompt = 'Now the movie end.'
    input_prompt = start_prompt + input_prompt + end_prompt
            
    qs = args.question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + input_prompt + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = input_prompt + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    print('> Input token num:', len(input_ids[0]))

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    cur_prompt = args.question
    with torch.inference_mode():
        model.update_prompt([[cur_prompt]])
        output_ids = model.generate(
            input_ids,
            images=video,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    print(outputs)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)