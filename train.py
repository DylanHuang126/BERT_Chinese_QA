import json
import torch
from tqdm import tqdm
import argparse
import pickle
import random
import numpy as np

from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig, BertPreTrainedModel
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

class Example():
    def __init__(
        self,
        qid,
        question_text,
        answer_text,
        context_text,
        start_pos,
        title,
        answerable,
        answers
    ):
        self.qid = qid
        self.question_text = question_text
        self.answer_text = answer_text
        self.context_text = context_text
        self.start_pos = start_pos
        self.title = title
        self.answerable = answerable
        self.answers = answers

class testExample():
    def __init__(
        self,
        qid,
        question_text,
        context_text,
    ):
        self.qid = qid
        self.question_text = question_text
        self.context_text = context_text
        
def process_examples(data, mode):
    examples = []
    for dat in tqdm(data['data']):
        title = dat['title']
        for paragraph in dat['paragraphs']:
            context_text = paragraph['context']
            for q in paragraph['qas']:
                qid = q['id']
                question_text = q['question']
                start_pos = None
                answer_text = None
                answers = []
                answerable = None
                
                if mode != 'testing':
                    answerable = q['answerable']
                    if answerable:
                        if mode == 'training':
                            answer = q['answers'][0]
                            answer_text = answer['text']
                            start_pos = answer['answer_start']
                        else:
                            answers = q['answers']
                
                    example = Example(
                        qid = qid,
                        question_text = question_text,
                        answer_text = answer_text,
                        context_text = context_text,
                        start_pos = start_pos,
                        title = title,
                        answerable = answerable,
                        answers = answers,
                    )
                    
                else:
                    example = testExample(
                        qid = qid,
                        question_text = question_text,
                        context_text = context_text
                    )
                examples.append(example)
    return examples

from torch.utils.data import Dataset, DataLoader

from torch.utils.data import Dataset, DataLoader

class QADataset(Dataset):
    def __init__(self, examples, mode, tokenizer, max_length):
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.start_pos = []
        self.end_pos = []
        self.answerables = []
        self.qids = []
        self.mode = mode
        #training
        for ex in tqdm(examples):
            context_ids = tokenizer.encode(ex.context_text, add_special_tokens=False)
            question_ids = tokenizer.encode(ex.question_text, add_special_tokens=False)
            inputs = tokenizer.prepare_for_model(context_ids, question_ids, max_length=max_length, truncation_strategy='only_first', pad_to_max_length=True)
            ans_span = []
            if self.mode == 'training':
                # if no answer in MAX_LENGTH, drop this sample
                if ex.answerable:
                    answer_ids = tokenizer.encode(ex.answer_text, add_special_tokens=False)
                    ans_span = self.find_ans_span(answer_ids, inputs['input_ids'])
                if len(ans_span) != 0:
                    self.start_pos.append(torch.tensor(ans_span[0][0], dtype=torch.long))
                    self.end_pos.append(torch.tensor(ans_span[0][1], dtype=torch.long))
                    self.input_ids.append(torch.tensor(inputs['input_ids']))
                    self.token_type_ids.append(torch.tensor(inputs['token_type_ids']))
                    self.attention_mask.append(torch.tensor(inputs['attention_mask']))
                    self.answerables.append(torch.tensor(ex.answerable, dtype=torch.float32))
                    self.qids.append(ex.qid)
                    
                else:
                    if not ex.answerable:
                        self.start_pos.append(torch.tensor(0, dtype=torch.long))
                        self.end_pos.append(torch.tensor(0, dtype=torch.long))
                        self.input_ids.append(torch.tensor(inputs['input_ids']))
                        self.token_type_ids.append(torch.tensor(inputs['token_type_ids']))
                        self.attention_mask.append(torch.tensor(inputs['attention_mask']))
                        self.answerables.append(torch.tensor(ex.answerable, dtype=torch.float32))
                        self.qids.append(ex.qid)
            else:
                self.input_ids.append(torch.tensor(inputs['input_ids']))
                self.token_type_ids.append(torch.tensor(inputs['token_type_ids']))
                self.attention_mask.append(torch.tensor(inputs['attention_mask']))
                self.qids.append(ex.qid)
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        if self.mode == 'training':
            return {
                'input_ids':self.input_ids[idx],
                'token_type_ids':self.token_type_ids[idx],
                'attention_mask':self.attention_mask[idx],
                'start_pos':self.start_pos[idx],
                'end_pos':self.end_pos[idx],
                'answerables':self.answerables[idx],
                'qid':self.qids[idx]
            }
        else:
            return {
                'input_ids':self.input_ids[idx],
                'token_type_ids':self.token_type_ids[idx],
                'attention_mask':self.attention_mask[idx],
                'qid':self.qids[idx],
            }
        
    def find_ans_span(self, sl, l):
        results=[]
        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                results.append((ind,ind+sll-1))
        return results

class QAmodel(BertPreTrainedModel):
    def __init__(self, config):
        super(QAmodel, self).__init__(config)
        self.bert = BertForQuestionAnswering(config).from_pretrained('bert-base-chinese', config=config)
    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        if start_positions is not None:
            loss, start_scores, end_scores, hiddens = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                                                start_positions=start_positions, end_positions=end_positions)
            return loss, start_scores, end_scores, hiddens[-1] #[batch, seq_length, hid_size]
        else:
            start_scores, end_scores, hiddens = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                                          start_positions=start_positions, end_positions=end_positions)
            return start_scores, end_scores, hiddens[-1]
    
def train(model, train_data, valid_data, args, pt='QA.pt'):
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(params = model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    min_val_loss = 10000
    
    now = datetime.datetime.now()
    writer = SummaryWriter(f'{args.log_dir}/{pt}-{now.month:02}{now.day:02}-{now.hour:02}{now.minute:02}')
    iters = 0
    for epoch in range(args.epochs):
        tr_loss = 0.
        step_iterator = tqdm(train_loader, desc='Training')
        model.train()
        if epoch == 1:
            for param in list(model.bert.bert.embeddings.parameters()):
                param.requires_grad = False
            for param in list(model.bert.bert.encoder.layer[0].parameters()):
                param.requires_grad = False
            optimizer = torch.optim.AdamW(params = model.parameters(), lr=args.freeze_lr)
        for step, batch in enumerate(step_iterator):
            iters += 1
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_pos = batch['start_pos'].to(device)
            end_pos = batch['end_pos'].to(device)
            answerables = batch['answerables'].to(device)  

            loss, start_score, enc_score, hidden = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                                         start_positions=start_pos, end_positions=end_pos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu().item()
            tr_loss += loss
            writer.add_scalar('train_loss', loss, iters)
            if step > 0 and step % 3000 == 0:
                val_loss = evaluate(model, valid_data, args)
                writer.add_scalar('val_loss', val_loss, iters)
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(model.state_dict(), pt)
                model.train()
        
        tr_loss /= step + 1
        writer.add_scalar('avg_train_loss', tr_loss, iters)
        print(f'[Epoch {epoch+1}] loss: {tr_loss:.3f}' + ', ' + f'val_loss: {val_loss:.3f}', flush=True)
    return

def evaluate(model, dataset, args):
    val_loader = DataLoader(dataset, batch_size=args.train_batch_size)
    model.eval()
    val_loss = 0.
    with torch.no_grad():
        step_iterator = tqdm(val_loader, desc='Evaluating')
        for step, batch in enumerate(step_iterator):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_pos = batch['start_pos'].to(device)
            end_pos = batch['end_pos'].to(device)
            answerables = batch['answerables'].to(device)
            loss, start_pos, end_pos, hidden = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=start_pos, end_positions=end_pos)
            val_loss += loss.detach().cpu().item()
        val_loss /= step + 1
        
    return val_loss


def plot_ans_dist(train_example):
    x = []
    for ex in tqdm(train_example):
        if ex.answerable:
            answer_ids = tokenizer.encode(ex.answer_text, add_special_tokens=False)
            x.append(len(answer_ids))
    
    fig, ax = plt.subplots(figsize=(20, 10))
    n, bins, patches = ax.hist(x, 30, density=True, cumulative=True)
    
    ax.set_title('Cumulative Answer Length')
    ax.set_xlabel('Answer Length')
    ax.set_ylabel('likelihood of occurence')

    plt.show()
    return

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return

if __name__ == '__main__':
    
    from tensorboardX import SummaryWriter
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import optim
    import datetime
    import matplotlib.pyplot as plt
    
    with open('data/train.json', 'r') as f:
        train_json = json.load(f)
    with open('data/dev.json', 'r') as f:
        valid_json = json.load(f)
    
    train_example = process_examples(train_json, 'training')
    valid_example = process_examples(valid_json, 'training')
    
    MAX_LENGTH = 512
    
    train_data = QADataset(train_example, 'training', tokenizer, MAX_LENGTH)
    valid_data = QADataset(valid_example, 'training', tokenizer, MAX_LENGTH)
    
    config = BertConfig.from_pretrained('bert-base-chinese', output_hidden_states=True)
    
    args = argparse.Namespace()
    args.train_batch_size = 4
    args.test_batch_size = 1
    args.lr = 8e-6
    args.freeze_lr = 8e-7
    args.log_dir = 'train_log'
    args.epochs = 2
    args.seed = 42
    args.max_grad_norm = 1.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QAmodel(config).to(device)
    set_seed(args.seed)
    
    train(model, train_data, valid_data, args, 'QA.pt')
    
    plot_ans_dist(train_example)
    
    