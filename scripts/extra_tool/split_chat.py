import json
import random
import argparse



def split_chat(input_path, output_path, max_round, media_token='<image>', img_token='<image>'):
    
    raw_data = json.load(open(input_path, 'r'))
    new_data = []
    print("Splitting data...")
    
    for _i, _data in enumerate(raw_data):
        if _i % 5000 == 0:
            print(f"Processing {_i} / {len(raw_data)}")
        
        if ('image' in _data or 'video' in _data):
            for _idx, _conv in enumerate(_data['conversations']):
                if _idx % (2 * max_round) == 0:
                    assert _conv['from'] == 'human', 'The first sentence should be from human'
                    if 'video' in _data:
                        conv_temp = {'id':_data['id'],
                                     'video': _data['video'],
                                     'conversations':[]}
                    else:
                        conv_temp = {'id':_data['id'],
                                     'image': _data['image'],
                                     'conversations':[]}
                    
                    if media_token not in _conv['value']:
                        # randomly add image token to the beginning or end of the sentence
                        if random.randint(0,1)==0:
                            _conv["value"] = img_token + '\n' + _conv["value"]
                        else:
                            _conv["value"] = _conv["value"] + '\n' + img_token
                    else:
                         _conv["value"] = _conv["value"].replace(media_token, img_token)
                         
                    conv_temp['conversations'].append(_conv) 
                else:
                    if media_token in _conv["value"]:
                        _conv["value"] = _conv["value"].replace(media_token, img_token)
                    
                    conv_temp['conversations'].append(_conv)
                    
                if len(conv_temp['conversations']) == 2 * max_round or _idx == len(_data['conversations'])-1:
                    new_data.append(conv_temp)
        else:
            new_data.append(_data)
    print(f"Total {len(new_data)} conversations")
    print("Saving data...")
    json.dump(new_data, open(output_path, 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/processed_data/processed_chat.json')
    parser.add_argument('--output', type=str, default='data/processed_data/processed_chat.json')
    parser.add_argument('--max-round', type=int, default=4)
    parser.add_argument('--media-token', type=str, default='<image>')
    args = parser.parse_args()
    
    split_chat(args.input, args.output, args.max_round, args.media_token)