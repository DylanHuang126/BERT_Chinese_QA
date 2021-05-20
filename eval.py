from tqdm import tqdm
import numpy as np
import random
import json
from argparse import ArgumentParser
from pathlib import Path

from train import *

parser = ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args = parser.parse_args()

with open(args.test_data_path, 'r') as f:
    test_data = json.load(f)
    
test_example = process_examples(test_data, 'testing')
MAX_LENGTH = 512
test_data = QADataset(test_example, 'predicting', tokenizer, MAX_LENGTH)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = BertConfig.from_pretrained('bert-base-chinese', output_hidden_states=True)
model = QAmodel(config)
model.load_state_dict(torch.load('./model/QA.pt'))
model.to(device)
model.eval()

def predict(model, dataset, tokenizer):
    test_loader = DataLoader(dataset, batch_size=1)
    model.eval()
    
    prediction = {}
    with torch.no_grad():
        step_iterator = tqdm(test_loader, desc='Predicting')
        for step, batch in enumerate(step_iterator):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_pos = end_pos = None
            start_scores, end_scores, hidden = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                                     start_positions=start_pos, end_positions=end_pos)
            
            start = torch.argmax(start_scores)
            end = torch.argmax(end_scores)
            
            if end == 0:
                prediction[batch['qid'][0]] = ""
            else:
                all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                if end - start > 30 or start > end:
                    _, idx_e = torch.topk(end_scores, 2)
                    _, idx_s = torch.topk(start_scores, 2)
                    idx = torch.cat((idx_s.squeeze(0), idx_e.squeeze(0)))
                    sorted_idx, idxx = torch.sort(idx)
                    
                    min_len = 1000
                    s_i = 0
                    for i in range(0, len(sorted_idx)-1):
                        lens = sorted_idx[i+1] - sorted_idx[i]
                        if lens > 0 and lens < min_len:
                            min_len = lens
                            s_i = i
                    answer = ''.join(all_tokens[sorted_idx[s_i] : sorted_idx[s_i+1]+1])
                else:
                    answer = ''.join(all_tokens[start : end+1])
                
                answer = answer.replace('##', '')
                if answer.find('[CLS]') != -1:
                    answer = answer.replace('[CLS]', '')
                if answer.find('[UNK]') != -1:
                    answer = answer.replace('[UNK]', '')
                if answer.find('《') != -1 and answer[-1] != '》':
                    answer = answer.replace('《', '')
                prediction[batch['qid'][0]] = answer
                         
    return prediction

prediction = predict(model, test_data, tokenizer)

with open(args.output_path, 'w') as f:
    json.dump(prediction, f)

