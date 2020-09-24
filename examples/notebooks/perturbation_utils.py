import torch
from transformers import *
import numpy as np
import sys, os
import random, math
import nltk


TOP_K=0.5

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_function_name_dict():
    # This is a better way than using eval() or globals()
    perturbation_name_func = {
        'sorted' : sort_sent,
        'reversed' : reverse_sent,
        'shuffled' : shuffle_sent,
        'empty' : empty_sent,
        'copy_premise_sort' : sort_sent,
        'copy_premise_reverse' : reverse_sent,
        'copy_premise_shuffle' : shuffle_sent,
        'most_important_word_in_premise' : most_important_word_in_premise,
        'keep_most_important_words' : keep_most_important_words,
        'repeat_most_important_words': repeat_most_important_words,
        'replace_least_important_words' : replace_least_important_words,
        'keep_one_word': keep_one_word,
    }

    return perturbation_name_func


def load_examples(filename, input_columns, is_pair=True):
    examples = []
    header = ''
    skipped = 0
    with open(filename, 'r') as fp:
        for i, line in enumerate(fp):
            if i ==0:
                header = line.strip()
                # Header line
                continue
            inp = []
            line_split = line.strip().split('\t')
            if max(input_columns) >= len(line_split):
                skipped +=1
                continue
            for col in list(input_columns):
                inp.append(line_split[col])
            examples.append((line_split, tuple(inp)))

    print('Number of examples loaded from file {} = {}'.format(filename, len(examples)))
    print('Number of examples skipped from file {} = {}'.format(filename, skipped))

    return examples, header

def reverse_sent(sent, **kwargs):

    max_tries = 10
    tries = 0

    while True:
        tok = nltk.word_tokenize(sent.lower())
        if tok[-1] == '.':
            last = tok[-1]
            new_tok = tok[:-1]
            new_tok = new_tok[::-1] # Reverse
            new_tok = new_tok + [last]
        else:
            new_tok = tok
            new_tok = new_tok[::-1] # Reverse
        new_sent = ' '.join(new_tok )
        if new_sent != sent:
            break
        elif tries >= max_tries:
            break
        else:
            tries +=1
            continue

    return new_sent

def empty_sent(sent, **kwargs):

    if 'replace_word' not in kwargs:
        replace_word = 'the'
    if 'replace_all' not in kwargs:
        replace_all = True

    max_tries = 10
    tries = 0

    while True:
        tok = sent.lower().split()
        if tok[-1] == '.':
            last = tok[-1]
            if replace_all:
                new_tok = [replace_word] * len(tok[:-1])
            else:
                new_tok = [replace_word]
            new_tok = new_tok + [last]
        else:
            if replace_all:
                new_tok = [replace_word] * len(tok)
            else:
                new_tok = [replace_word]
        new_sent = ' '.join(new_tok )
        if new_sent != sent:
            break
        elif tries >= max_tries:
            break
        else:
            tries +=1
            continue

    return new_sent


def shuffle_sent(sent):

    max_tries = 10
    tries = 0

    while True:
        tok = nltk.word_tokenize(sent.lower())
        if tok[-1] == '.':
            last = tok[-1]
            new_tok = tok[:-1]
            random.shuffle(new_tok)
            new_tok = new_tok + [last]
        else:
            random.shuffle(tok)
            new_tok = tok
        new_sent = ' '.join(new_tok )
        if new_sent != sent:
            break
        elif tries >= max_tries:
            break
        else:
            tries +=1
            continue

    return new_sent


def sort_sent(sent):

    max_tries = 10
    tries = 0

    while True:
        tok = nltk.word_tokenize(sent.lower())
        if tok[-1] == '.':
            last = tok[-1]
            new_tok = tok[:-1]
            new_tok.sort() # Sort step
            new_tok = new_tok + [last]
        else:
            new_tok = tok
            new_tok.sort() # Sort step
        new_sent = ' '.join(new_tok )
        if new_sent != sent:
            break
        elif tries >= max_tries:
            break
        else:
            tries +=1
            continue

    return new_sent

def simple_perturbation(examples, is_pair, on_sent2, input_columns, func, **kwargs):
    """
    For perturbations that do not require copy from the other sent like 'copy_premise_sort'
    """
    new_examples = []

    for ex in examples:
        new_cols = ex[0]
        if is_pair:
            if on_sent2:
                if len(ex[0][input_columns[1]]) < 5:
                    continue
                new_cols[input_columns[1]] = func(ex[0][input_columns[1]])
            else:
                if len(ex[0][input_columns[0]]) < 5:
                    continue
                new_cols[input_columns[0]] = func(ex[0][input_columns[0]])
        else:
            new_cols[input_columns[0]] = func(ex[0][input_columns[0]], **kwargs)
        new_examples.append(new_cols)

    return new_examples

def copy_perturbation(examples, is_pair, on_sent2, input_columns, func, **kwargs):
    """
    For perturbations that require copy from the other sent like 'copy_premise_sort'
    Not Defined for single sentence tasks
    """
    new_examples = []

    for ex in examples:
        new_cols = ex[0]
        if is_pair:
            if on_sent2:
                # For on_sent2=True, copy sent1 in sent2 and then apply another perturbation (like sort)
                if len(ex[0][input_columns[0]]) < 5:
                    continue
                new_cols[input_columns[1]] = func(ex[0][input_columns[0]], **kwargs)
            else:
                # For on_sent2=False, copy sent2 in sent1 and then apply another perturbation (like sort)
                if len(ex[0][input_columns[1]]) < 5:
                    continue
                new_cols[input_columns[0]] = func(ex[0][input_columns[1]], **kwargs)
        else:
            new_cols[input_columns[0]] = func(ex[0][input_columns[0]], **kwargs)
        new_examples.append(new_cols)

    return new_examples

def write_new_examples_to_file(filename, examples, header=None):

    fp = open(filename, 'w')
    if header is not None:
        fp.write(header + '\n')

    for ex in examples:
        output_str = '\t'.join(ex)
        fp.write(output_str + '\n')

    fp.close()

def generate_perturbation_simple(filename, task, input_dir, output_dir, input_columns, perturbation_func, output_filename=None, is_pair=True, on_sent2=True, **kwargs):
    """
    examples : List of raw text examples
    is_pair : Is it a sentence pair task
    on_sent2 : If a sentence pair task,
                apply on sentence 2. If False,
                apply on sentence 1
    kwargs : To provide args to each pertrubation funcs
    """

    examples, header = load_examples(input_dir + '/' + filename, input_columns, is_pair=is_pair)

    # See if output_dir exists, otherwise create it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if output_filename is None:
        # assume file is tsv
        output_filename = task + '_' + filename[:-4] + '_' + perturbation_func + '.tsv'

        # If is_pair task, then change filename to include arg on_sent2
        if is_pair:
            if on_sent2:
                output_filename = task + '_' + filename[:-4] + '_' + perturbation_func + '_sent2' + '.tsv'
            else:
                output_filename = task + '_' + filename[:-4] + '_' + perturbation_func + '_sent1' + '.tsv'

    if os.path.isfile(output_dir + '/' + output_filename):
        print('Perturbation File {} already exists !! To override, please delete that file.'.format(output_dir + '/' + output_filename))
        return output_filename

    func = get_function_name_dict()[perturbation_func]

    if 'copy' in perturbation_func:
        if is_pair:
            new_examples = copy_perturbation(examples, is_pair, on_sent2, input_columns, func, **kwargs)
        else:
            print('Copy pertrubation not valid for single sentence tasks. Return None')
            return None
    else:
        new_examples = simple_perturbation(examples, is_pair, on_sent2, input_columns, func, **kwargs)


    write_new_examples_to_file(output_dir + '/' + output_filename, new_examples, header)
    print(len(new_examples))
    print('Successfully generated perturbations for funcion : {} with params (is_pair={}, on_sent2={}) saved to file {}'.format(
                                                                        perturbation_func, str(is_pair), str(on_sent2), output_filename))

    return output_filename


def get_gradients(args, model, batch):
    """
    Get gradients by applying backward hook
    """

    extracted_grads = {}

    model.eval()

    def extract_grad_hook(module, grad_in, grad_out):
        extracted_grads['embed'] = grad_out[0]


    def add_hooks(bert_model):
        module = bert_model.embeddings.word_embeddings
        module.register_backward_hook(extract_grad_hook)
    def add_hooks_bart(bart_model):
        module = bart_model.shared
        module.register_backward_hook(extract_grad_hook)
    def add_hooks_xlnet(xlnet_model):
        module = xlnet_model.word_embedding
        module.register_backward_hook(extract_grad_hook)

    # Add hooks to bert embeddings
    # For XLNet, we are interested in  model.transformer.word_embedding
    # For Electra, we are interested in model.electra.embeddings.word_embeddings
    # For Bart, we are interested in model.model.shared
    if args.model_type == 'roberta':
        add_hooks(model.roberta)
    elif args.model_type == 'bert':
        add_hooks(model.bert)
    elif args.model_type.lower() == 'electra':
        add_hooks(model.electra)
    elif args.model_type == 'bart':
        add_hooks_bart(model.model)
    elif args.model_type.lower() == 'xlnet':
        add_hooks_xlnet(model.transformer)
    else:
        raise NotImplementedError

    #print(batch.keys())
    #batch = tuple(t.to(args.device) for t in batch)
    batch_device = {k: v.to(args.device) for k,v in batch.items() if k not in ['input_pair_lengths']}
    inputs = batch_device


    # Take care of mems attribute that XLNet uses TODO
    #inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
    #if args.model_type != "distilbert":
    #    inputs["token_type_ids"] = (
    #        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
    #    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

    outputs = model(**inputs)
    #has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])
    #if has_labels:
    loss, logits = outputs[:2]

    loss.backward()

    if args.model_type.lower() == 'xlnet':
        return extracted_grads['embed'].permute(1,0,2)

    return extracted_grads['embed']

def get_gradients_last_layer(args, model, batch):
    """
    Get gradients by applying backward hook
    """

    extracted_grads = {}

    model.eval()

    def extract_grad_hook(module, grad_in, grad_out):
        extracted_grads['encoder'] = grad_out[0]


    def add_hooks(bert_model):
        module = bert_model.encoder
        module.register_backward_hook(extract_grad_hook)

    # Add hooks to bert embeddings
    if args.model_type == 'roberta':
        add_hooks(model.roberta)
    elif args.model_type == 'bert':
        add_hooks(model.bert)
    else:
        raise NotImplementedError

    batch = tuple(t.to(args.device) for t in batch)


    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
    if args.model_type != "distilbert":
        inputs["token_type_ids"] = (
            batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

    outputs = model(**inputs)

    loss, logits = outputs[:2]

    loss.backward()

    return extracted_grads['encoder']


def get_importance_order(args, model, grads, batch, is_pair=True):

    # Returns indices sorted by their importance (most - > least)

    # Add hooks to bert embeddings
    # For XLNet, we are interested in  model.transformer.word_embedding
    # For Electra, we are interested in model.electra.embeddings.word_embeddings
    # For Bart, we are interested in model.model.shared
    if args.model_type == 'roberta':
        #embeds = model.roberta.embeddings.word_embeddings(batch[0].to(args.device))
        embeds = model.roberta.embeddings.word_embeddings(batch['input_ids'].to(args.device))
    elif args.model_type == 'bert':
        #embeds = model.bert.embeddings.word_embeddings(batch[0].to(args.device))
        embeds = model.bert.embeddings.word_embeddings(batch['input_ids'].to(args.device))
    elif args.model_type.lower() == 'electra':
        embeds = model.electra.embeddings.word_embeddings(batch['input_ids'].to(args.device))
    elif args.model_type == 'bart':
        embeds = model.model.shared(batch['input_ids'].to(args.device))
    elif args.model_type.lower() == 'xlnet':
        embeds = model.transformer.word_embedding(batch['input_ids'].to(args.device))
    else:
        raise NotImplementedError



    #print(embeds.shape, grads.shape)
    #print(batch['input_ids'].shape)
    #print(batch['input_ids'])
    one_hot_grad = -torch.mul(embeds, grads)
    #one_hot_grad = torch.mul(embeds, grads)

    if not is_pair:
        importance_order_bert = []
        bert_tok = []
        bert_grads = []
        #for i in range(batch[0].shape[0]):
        for i in range(batch['input_ids'].shape[0]):
            lengths = batch['input_pair_lengths'][i].item()
            grads_ = one_hot_grad[i][1:1+lengths]
            bert_tok.append(batch['input_ids'][i][1:1 + lengths])
            bert_grads.append(grads_)
            order = np.argsort(grads_.sum(-1).data.cpu().numpy())
            importance_order_bert.append(list(order))

        return importance_order_bert, bert_tok, bert_grads


    importance_order_bert1 = []
    importance_order_bert2 = []
    bert1_tok = []
    bert2_tok = []
    bert1_grads = []
    bert2_grads = []


    for i in range(batch['input_ids'].shape[0]):
        lengths = batch['input_pair_lengths'][i]
        bert1 = batch['input_ids'][i][1:1 + lengths[0]]
        bert2 = batch['input_ids'][i][lengths[0] + 3 : lengths[0] + 3 + lengths[1]]
        grads1 = one_hot_grad[i][1:1 + lengths[0]]
        grads2 = one_hot_grad[i][lengths[0] + 3 : lengths[0] + 3 + lengths[1]]
        bert1_tok.append(bert1)
        bert2_tok.append(bert2)
        bert1_grads.append(grads1)
        bert2_grads.append(grads2)
        order = np.argsort(grads1.sum(-1).data.cpu().numpy())
        importance_order_bert1.append(list(order))
        order = np.argsort(grads2.sum(-1).data.cpu().numpy())
        importance_order_bert2.append(list(order))

    return importance_order_bert1, importance_order_bert2, bert1_tok, bert2_tok, bert1_grads, bert2_grads

def keep_most_important_words(importance_order, bert_ids, special_ids, **kwargs):

    if 'topk' not in kwargs:
        kwargs['topk'] = TOP_K

    # importance_order gives index, not the actual token_ids

    new_bert = []
    importance_order_ids = []

    for idx in importance_order:
        if bert_ids.tolist()[idx] not in special_ids:
            importance_order_ids.append(bert_ids.tolist()[idx])

    to_keep = math.ceil(len(importance_order_ids) * kwargs['topk'])
    to_keep_ids = importance_order_ids[:to_keep]


    toks = bert_ids.tolist()
    new_toks = []

    keep_ids = []
    keep_ids.extend(special_ids)
    keep_ids.extend(to_keep_ids)
    for t in toks:
        if t not in keep_ids:
            continue
        else:
            new_toks.append(t)
    new_bert.append(new_toks)

    # Wont return tensor, because we can have different length sequences
    return new_bert

def keep_one_word(importance_order, bert_ids, special_ids, **kwargs):

    # importance_order gives index, not the actual token_ids

    if 'num_words' not in kwargs:
        kwargs['num_words'] = 1

    new_bert = []
    importance_order_ids = []

    for idx in importance_order:
        if bert_ids.tolist()[idx] not in special_ids:
            importance_order_ids.append(bert_ids.tolist()[idx])

    to_keep_ids = importance_order_ids[:kwargs['num_words']]


    toks = bert_ids.tolist()
    new_toks = []

    keep_ids = []
    keep_ids.extend(special_ids)
    keep_ids.extend(to_keep_ids)
    for t in toks:
        if t not in keep_ids:
            continue
        else:
            new_toks.append(t)
    new_bert.append(new_toks)

    # Wont return tensor, because we can have different length sequences
    return new_bert


def repeat_most_important_words(importance_order, bert_ids, special_ids, **kwargs):
    # importance_order gives index, not the actual token_ids
    if 'topk' not in kwargs:
        kwargs['topk'] = TOP_K

    new_bert = []

    importance_order_ids = []

    for idx in importance_order:
        if bert_ids.tolist()[idx] not in special_ids:
            importance_order_ids.append(bert_ids.tolist()[idx])

    to_keep = math.ceil(len(importance_order_ids) * kwargs['topk'])
    to_keep_ids = importance_order_ids[:to_keep]

    toks = bert_ids.tolist()
    new_toks = []

    keep_ids = []
    keep_ids.extend(special_ids)
    keep_ids.extend(to_keep_ids)
    for t in toks:
        if t not in keep_ids:
            new_toks.append(random.choice(to_keep_ids))
        else:
            new_toks.append(t)
    new_bert.append(new_toks)

    return new_bert

def replace_least_important_words(importance_order, bert_ids, special_ids, **kwargs):
    # importance_order gives index, not the actual token_ids
    if 'topk' not in kwargs:
        kwargs['topk'] = TOP_K

    if 'replace_token_ids' not in kwargs:
        raise Exception

    replace_token_ids = kwargs['replace_token_ids']

    new_bert = []
    importance_order_ids = []

    for idx in importance_order:
        if bert_ids.tolist()[idx] not in special_ids:
            importance_order_ids.append(bert_ids.tolist()[idx])

    to_keep = math.ceil(len(importance_order_ids) * kwargs['topk'])
    to_keep_ids = importance_order_ids[:to_keep]

    toks = bert_ids.tolist()
    new_toks = []

    keep_ids = []
    keep_ids.extend(special_ids)
    keep_ids.extend(to_keep_ids)
    for t in toks:
        if t not in keep_ids:
            new_toks.append(random.choice(replace_token_ids))
        else:
            new_toks.append(t)
    new_bert.append(new_toks)

    return new_bert

def most_important_word_in_premise(importance_order, bert_ids1, bert_ids2, special_ids, **kwargs):
    # importance_order of premise (sent 1)
    new_bert = []
    importance_order_ids = []

    for idx in importance_order:
        if bert_ids1.tolist()[idx] not in special_ids:
            importance_order_ids.append(bert_ids1.tolist()[idx])

    most_important_tok = importance_order_ids[0]

    toks = bert_ids2.tolist()
    new_toks = []

    keep_ids = []
    keep_ids.extend(special_ids)

    for t in toks:
        if t not in keep_ids:
            new_toks.append(most_important_tok)
        else:
            new_toks.append(t)
    new_bert.append(new_toks)

    return new_bert


def generate_perturbation_gradient(dataloader, tokenizer, model, args, filename, task, input_dir, output_dir, input_columns, perturbation_func, output_filename=None, is_pair=True, on_sent2=True, **kwargs):

    examples, header = load_examples(input_dir + '/' + filename, input_columns, is_pair=is_pair)

    # See if output_dir exists, otherwise create it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if 'replace_token_ids' not in kwargs:
        kwargs['replace_token_ids'] = list(range(tokenizer.vocab_size))[100:20000]

    if output_filename is None:
        # assume file is tsv
        if 'topk' not in kwargs:
            kwargs['topk'] = TOP_K
        output_filename = task + '_' + filename[:-4] + '_' + perturbation_func +  '_' + str(kwargs['topk']) + '.tsv'

        # If is_pair task, then change filename to include arg on_sent2
        if is_pair:
            if on_sent2:
                if 'topk' not in kwargs:
                    kwargs['topk'] = TOP_K
                output_filename = task + '_' + filename[:-4] + '_' + perturbation_func + '_sent2' +  '_' + str(kwargs['topk']) + '.tsv'
            else:
                if 'topk' not in kwargs:
                    kwargs['topk'] = TOP_K
                output_filename = task + '_' + filename[:-4] + '_' + perturbation_func + '_sent1' +  '_' + str(kwargs['topk']) + '.tsv'

    if os.path.isfile(output_dir + '/' + output_filename):
        print('Perturbation File {} already exists !! To override, please delete that file.'.format(output_dir + '/' + output_filename))
        return output_filename

    func = get_function_name_dict()[perturbation_func]

    new_examples = []
    overall_count = 0

    if not is_pair and perturbation_func == 'most_important_word_in_premise':
        print('Perturbation : {} Invalid for single sentence. Return None '.format(perturbation_func))
        return None

    for i in range(len(dataloader)):
        if i %100 == 0:
            print('Done with {0}/{1} batches'.format(i, len(dataloader)))
       # print('Done with {0}/{1} batches'.format(i, len(dataloader)))

        batch = list(dataloader)[i]
        grads = get_gradients(args, model, batch)

        if not is_pair:
            # If a single sentence task like SST-2
            importance_order_bert, bert_tok, bert_grads = get_importance_order(args, model, grads, batch, is_pair=is_pair)
            for j in range(len(importance_order_bert)):
                ex = examples[overall_count]
                new_bert_ids = func(importance_order_bert[j], bert_tok[j],
                                                        tokenizer.all_special_ids, **kwargs)[0]
                new_sent = tokenizer.decode(new_bert_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                new_cols = ex[0]
                new_cols[input_columns[0]] = new_sent
                new_examples.append(new_cols)
                overall_count +=1

        else:
            #print(len(examples))
            importance_order_bert1, importance_order_bert2, bert1_tok, bert2_tok, _, _ = get_importance_order(args, model, grads, batch, is_pair=is_pair)
            for j in range(len(importance_order_bert2)):
                #print(overall_count)
                ex = examples[overall_count]
                new_cols = ex[0]
                if on_sent2:
                    # For on_sent2=True, copy sent1 in sent2 and then apply another perturbation (like sort)
                    if perturbation_func == 'most_important_word_in_premise':
                        new_bert_ids = func(importance_order_bert1[j], bert1_tok[j], bert2_tok[j],
                                            tokenizer.all_special_ids, **kwargs)[0]
                    else:
                        new_bert_ids = func(importance_order_bert2[j], bert2_tok[j],
                                                        tokenizer.all_special_ids, **kwargs)[0]
                    new_sent = tokenizer.decode(new_bert_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    new_cols[input_columns[1]] = new_sent
                else:
                    # For on_sent2=False, copy sent2 in sent1 and then apply another perturbation (like sort)
                    if perturbation_func == 'most_important_word_in_premise':
                        new_bert_ids = func(importance_order_bert2[j], bert2_tok[j], bert1_tok[j],
                                            tokenizer.all_special_ids, **kwargs)[0]
                    else:
                        new_bert_ids = func(importance_order_bert1[j], bert1_tok[j],
                                                        tokenizer.all_special_ids, **kwargs)[0]
                    new_sent = tokenizer.decode(new_bert_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    new_cols[input_columns[0]] = new_sent
                new_examples.append(new_cols)
                overall_count +=1


    write_new_examples_to_file(output_dir + '/' + output_filename, new_examples, header)
    print(len(new_examples))
    print('Successfully generated perturbations for funcion : {} with params (is_pair={}, on_sent2={}) saved to file {}'.format(
                                                                        perturbation_func, str(is_pair), str(on_sent2), output_filename))

    return output_filename
