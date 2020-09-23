import dataclasses
import logging
import os
import sys, random
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from transformers.data.data_collator import DataCollator, default_data_collator

from examples.notebooks import perturbation_utils


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

if (
    os.path.exists(training_args.output_dir)
    and os.listdir(training_args.output_dir)
    and training_args.do_train
    and not training_args.overwrite_output_dir
):
    raise ValueError(
        f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
    )

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
)
logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    training_args.local_rank,
    training_args.device,
    training_args.n_gpu,
    bool(training_args.local_rank != -1),
    training_args.fp16,
)
logger.info("Training/evaluation parameters %s", training_args)

def set_seed(training_args):
    perturbation_utils.set_seed(training_args)
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    if training_args.n_gpu > 0:
        torch.cuda.manual_seed_all(training_args.seed)

# Set seed
set_seed(training_args)

try:
    num_labels = glue_tasks_num_labels[data_args.task_name]
    output_mode = glue_output_modes[data_args.task_name]
except KeyError:
    raise ValueError("Task not found: %s" % (data_args.task_name))


# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.

config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
    criterion=training_args.criterion,
    label_smoothing=training_args.label_smoothing,
    focal_gamma=training_args.focal_gamma
)

print(config)
print(config.model_type)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
)

model.to(training_args.device)


# Load dataset
eval_dataset = (
    GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir, return_input_pair_lengths=True)
    if training_args.do_eval
    else None
)


def get_eval_dataloader(args, eval_dataset):
    """
    Returns the evaluation :class:`~torch.utils.data.DataLoader`.

    Args:
        eval_dataset (:obj:`Dataset`, `optional`):
            If provided, will override `self.eval_dataset`.
    """
    if eval_dataset is None :
        raise ValueError("Trainer: evaluation requires an eval_dataset.")

    #eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

    sampler = SequentialSampler(eval_dataset)

    data_loader = DataLoader(
        eval_dataset,
        sampler=sampler,
        batch_size=args.eval_batch_size,
        collate_fn=default_data_collator,
        drop_last=args.dataloader_drop_last,
    )

    return data_loader

eval_dataloader = get_eval_dataloader(training_args, eval_dataset)

model.eval()

# Code to generate perturbation functions
pair_tasks = {
    'sst-2': False,
    'snli' : True,
    'smnli' : True,
    'mnli' : True,
    'qnli' : True,
    'xnli' : True,
    'qqp' : True,
    'mrpc' : True,
    'wnli' : True,
    'rte': True
}

# Dictionary of perturbation functions, vals : if they are not valid for single sentence
# True means valid only for sentence pair tasks
# Here for convenience, we assume premise refers to sentence 1, hypo refers to sentence 2
perturbation_functions = {
        'sorted' : ('simple', False),
        'reversed' : ('simple', False),
        'shuffled' : ('simple', False),
        'empty' : ('simple', True),
        'copy_premise_sort' : ('simple', True),
        'keep_most_important_words' : ('grad', False),
        'most_important_word_in_premise' : ('grad', True),
        'repeat_most_important_words': ('grad', False),
        'replace_least_important_words' : ('grad', False),
    }

#perturbation_functions = {
#        'sorted' : ('simple', False),
#        'reversed' : ('simple', False),
#        'shuffled' : ('simple', False),
#        'empty' : ('simple', True),
#        'copy_premise_sort' : ('simple', True),
#        'copy_premise_reverse' : ('simple', True),
#        'copy_premise_shuffle' : ('simple', True),
#        'most_important_word_in_premise' : ('grad', True),
#        'keep_most_important_words' : ('grad', False),
#        'repeat_most_important_words': ('grad', False),
#        'replace_least_important_words' : ('grad', False),
#        'keep_one_word': ('grad', False),
#    }



# For [sorted, reversed, 'shuffled', 'empty', 'copy_premise_*'], we need
# original sentence inputs


# So we have to specify the input file formats

# Keys are the tasks, vals are the column to read input from
# Ex: for qqp, question1 is column3 (indexed from 0)....
# Last value of tuple specifies the label column
# for qqp, val = (3, 4, 5)
input_format = {
    'sst-2': (0, 1),
    'snli' : (7, 8, -1),
    'smnli' : (7, 8, -1),
    'mnli' : (8, 9, -1),
    'qnli' : (1, 2, -1),
    'xnli' : (1, 2, -1), # Not in GLUE
    'qqp' : (3, 4, 5),
    'mrpc' : (3, 4, 0),
    'wnli' : (1, 2, -1),
    'rte': (1, 2, -1),
}


if training_args.perturbation != 'all':
    perturbations = {}
    for pert in training_args.perturbation.split(','):
        perturbations[pert] = perturbation_functions[pert]
else:
    perturbations = perturbation_functions


logger.info('Perturbation functions to evaluate : ' + str(training_args.perturbation))

data_dir = "data/" + data_args.task_name.upper() + '/'

generated_filenames = {}
for pert_fn in perturbations.keys():
    if perturbations[pert_fn][0] != 'simple':
        continue
    generated_filenames[pert_fn] = perturbation_utils.generate_perturbation_simple('dev.tsv', data_args.task_name.lower(),
                                    data_dir,
                                training_args.save_perturb_output_dir,
                                input_format[data_args.task_name.lower()], pert_fn, is_pair=pair_tasks[data_args.task_name.lower()], on_sent2=True)




print(generated_filenames)

kwargs = {}
topk_perts = ['keep_most_important_words', 'repeat_most_important_words', 'replace_least_important_words']
#topk_vals = [0.3, 0.5]
topk_vals = [0.5]
args = training_args
args.model_type = config.model_type

for pert_fn in perturbations.keys():
    if perturbations[pert_fn][0] != 'grad':
        continue
    if pert_fn in generated_filenames:
        continue
    if pert_fn in topk_perts:
        # Evaluate for each value of topk
        for topk in topk_vals:
            print(topk)
            kwargs['topk'] = topk
            generated_filenames[pert_fn] = perturbation_utils.generate_perturbation_gradient(eval_dataloader, tokenizer,
                                 model, args,
                                 'dev_matched.tsv', data_args.task_name.lower(), data_dir,
                                training_args.save_perturb_output_dir,
                                input_format[data_args.task_name.lower()], pert_fn,is_pair=pair_tasks[data_args.task_name.lower()], on_sent2=True, **kwargs)
    else:
        generated_filenames[pert_fn] = perturbation_utils.generate_perturbation_gradient(eval_dataloader, tokenizer,
                                 model, args,
                                 'dev_matched.tsv', data_args.task_name.lower(), data_dir,
                                training_args.save_perturb_output_dir,
                                input_format[data_args.task_name.lower()], pert_fn,is_pair=pair_tasks[data_args.task_name.lower()], on_sent2=True, **kwargs)


print(generated_filenames)

def forward_pass(model, batch, args, return_logits=True):

    model.eval()
    #batch = tuple(t.to(args.device) for t in batch)

    batch_device = {k: v.to(args.device) for k,v in batch.items() if k not in ['input_pair_lengths']}
    inputs = batch_device

    with torch.no_grad():
        #inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        #if args.model_type != "distilbert":
        #    inputs["token_type_ids"] = (
        #        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
        #    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

        outputs = model(**inputs)

        tmp_eval_loss, logits = outputs[:2]
        preds_batch = list(np.argmax(logits.detach().cpu().numpy(), axis=1))
        gold_batch = list(inputs["labels"].detach().cpu().numpy())

    if return_logits:
        return preds_batch, gold_batch, logits

    return preds_batch, gold_batch


from sklearn.metrics import confusion_matrix
def get_accuracy(all_pred_labels, all_gold_labels):
    num_correct = np.equal(np.array(all_pred_labels), np.array(all_gold_labels)).sum()
    if len(all_pred_labels) != 0:
        return float(num_correct)/len(all_pred_labels)

def entropy(log_score):
    score = np.exp(log_score)

    n_samples = score.shape[0]
    ent = np.multiply(score, log_score)

    ent = -np.sum(ent, axis=1)
    total_ent = np.sum(ent)
    if n_samples == 0:
        return 0
    else:
        return total_ent/n_samples


def average_confidence(log_probs):
    # Average of probs of maximum log score
    probs = np.exp(log_probs)
    max_probs = np.max(probs, axis=1)
    n_samples = probs.shape[0]
    if n_samples == 0:
        return 0
    else:
        return np.sum(max_probs)/n_samples

def confusion_mat(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def get_statistics(all_preds, all_golds, all_preds_perturbed, all_pred_probs,
                   all_pred_probs_perturbed, fp_log, fp_preds_perturbed):

    # all_preds, all_preds_perturbed, all_golds : list
    # all_pred_probs, all_pred_probs_perturbed : np array of log scores
    # Compute all statistics --
    # Average Entropy, Average Confidence
    # Average Accuracy
    # %Retained Predictions
    # %Predictions of each class
    # Confusion Matrix
    fp_log.write('--'*40 + '\n')
    fp_log.write('--'*40 + '\n')
    fp_log.write('--'*40 + '\n')
    fp_log.write('Calculating statistics' + '\n')

    original_acc = get_accuracy(all_preds, all_golds)
    perturb_acc = get_accuracy(all_preds_perturbed, all_golds)
    retained_preds = get_accuracy(all_preds_perturbed, all_preds)

    original_entropy = entropy(all_pred_probs)
    perturb_entropy = entropy(all_pred_probs_perturbed)

    original_confidence = average_confidence(all_pred_probs)
    perturb_confidence = average_confidence(all_pred_probs_perturbed)

    original_confusion_mat = confusion_mat(all_golds, all_preds)
    perturb_confusion_mat = confusion_mat(all_golds, all_preds_perturbed)

    num_classes = original_confusion_mat.shape[0]

    fp_log.write('Accuracy on original Dev Set : ' + str(original_acc) + '\n')
    fp_log.write('Accuracy on Perturbed Dev Set : ' + str(perturb_acc)+ '\n')
    fp_log.write('--'*40+ '\n')
    fp_log.write('Average Entropy on original Dev Set : ' + str(original_entropy) + '\n')
    fp_log.write('Average Entropy on Perturbed Dev Set : ' + str(perturb_entropy) + '\n')
    fp_log.write('--'*40 + '\n')
    fp_log.write('--'*40 + '\n')
    fp_log.write('Average Confidence on original Dev Set : ' + str(original_confidence) + '\n')
    fp_log.write('Average Confidence on Perturbed Dev Set : ' + str(perturb_confidence) + '\n')
    fp_log.write('--'*40 + '\n')
    fp_log.write('% Retained Predictions : ' + str(retained_preds) + '\n')
    fp_log.write('--'*40 + '\n')
    fp_log.write('Number of Predictions of each type in original dev file: ' + '\n')
    fp_log.write('Please refer to original dev file to see what each class number means.' + '\n')
    for i in range(num_classes):
        fp_log.write('For class {0}, Number of Predictions : {1}'.format(i, original_confusion_mat[:, i].sum()) + '\n')

    fp_log.write('Printing Confusion matrix for original dev file --- ' + '\n')
    fp_log.write('\n'.join([str(original_confusion_mat[i]) for i in range(original_confusion_mat.shape[0])]) + '\n')
    fp_log.write('Remember columns are predictions and rows are gold.' + '\n')

    fp_log.write('--'*40 + '\n')
    fp_log.write('Number of Predictions of each type in perturbed dev file: ' + '\n')
    fp_log.write('Please refer to original dev file to see what each class number means.' + '\n')
    for i in range(num_classes):
        fp_log.write('For class {0}, Number of Predictions : {1}'.format(i, perturb_confusion_mat[:, i].sum()) + '\n')
    fp_log.write('Printing Confusion matrix for perturbed dev file --- ' + '\n')
    fp_log.write('\n'.join([str(perturb_confusion_mat[i]) for i in range(perturb_confusion_mat.shape[0])]) + '\n')
    fp_log.write('Remember columns are predictions and rows are gold.' + '\n')

    fp_log.write('--'*40 + '\n')
    fp_log.write('--'*40 + '\n')
    fp_log.write('--'*40 + '\n')
    fp_log.write('Writing predictions ...' + '\n')
    fp_log.write('--'*40 + '\n')
    fp_log.close()

    fp_preds_perturbed.write('Gold' + '\t' + 'Original Prediction' + '\t' 'Perturbed Prediction' + '\t'
                            'Original Confidence' + '\t' + 'Perturbed Confidence' + '\n')
    probs = np.exp(all_pred_probs)
    max_probs = np.max(probs, axis=1)

    probs_perturb = np.exp(all_pred_probs_perturbed)
    max_probs_perturb = np.max(probs_perturb, axis=1)
    for i in range(len(all_preds)):
        fp_preds_perturbed.write(str(all_golds[i]) + '\t' + str(all_preds[i]) + '\t' + str(all_preds_perturbed[i])
                                 + '\t' + str(max_probs[i]) + '\t' + str(max_probs_perturb[i]) + '\n')

    fp_preds_perturbed.close()
    return


def do_temperature_scaling(logits, temp):
    temperature_ = torch.ones(1) * temp

    temperature = temperature_.unsqueeze(1).expand(logits.size(0), logits.size(1)).to(args.device)

    return logits / temperature


def evaluate_one_perturbation(args, training_args, filename, output_dir ):
    args.device = training_args.device

    log_dir = output_dir + '/logs'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_file = log_dir + '/' + 'log_' + filename[:-4] + '.txt'
    preds_perturbed_file = log_dir + '/' + 'preds_' + filename[:-4] + '.tsv'

    fp_log = open(log_file , 'w')
    fp_log.write(str(vars(args)) + '\n')
    fp_log.write('For perturbation : ' + str(filename) + '\n')
    fp_preds_perturbed = open(preds_perturbed_file, 'w')


    ### Evaluation code

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    #eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    #eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    #eval_task = eval_task_names[0]
    #eval_output_dir = eval_outputs_dirs[0]

    args.overwrite_cache = True

    data_dir = args.data_dir



    #eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True, evaluate_file_prefix='dev')
    eval_dataset = (
    GlueDataset(args, tokenizer=tokenizer, mode="dev", cache_dir=args.data_dir, return_input_pair_lengths=True)
)
    args.data_dir = output_dir
    eval_dataset_perturbed = (
    GlueDataset(args, tokenizer=tokenizer, mode="other", cache_dir=args.data_dir, return_input_pair_lengths=True,
                other_split_prefix=filename[:-4], other_split_filename=filename)
)
    #eval_dataset_perturbed = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True, evaluate_file_prefix=filename[:-4])
    args.data_dir = data_dir

    fp_log.write('--'*40 + '\n')
    fp_log.write('--'*40 + '\n')
    fp_log.write('--'*40 + '\n')
    fp_log.write('Successfully loaded examples!' + '\n')


    #args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                        collate_fn=default_data_collator)

    # Note that DistributedSampler samples randomly
    eval_sampler_perturbed = SequentialSampler(eval_dataset_perturbed)
    eval_dataloader_perturbed = DataLoader(eval_dataset_perturbed, sampler=eval_sampler_perturbed, batch_size=args.eval_batch_size,
                                        collate_fn=default_data_collator)


    fp_log.write(str(len(eval_dataloader)) + '\n')
    fp_log.write(str(len(eval_dataloader_perturbed)) + '\n')
    #cuda0 = torch.device('cuda:0')

    model.eval()

    #fp_log.write('Printing first example from files ' + '\n')
    #batch = list(eval_dataloader)[0]
    #batch_perturbed = list(eval_dataloader_perturbed)[0]
    #fp_log.write('From Dev file ' + '\n')
    #fp_log.write('Shape :' + str(batch_perturbed[0].shape) + '\n')
    #fp_log.write(tokenizer.decode(batch[0][1], skip_special_tokens=True, clean_up_tokenization_spaces=True) + '\n')
    #fp_log.write('From Perturbation file ' + '\n')
    #fp_log.write(tokenizer.decode(batch_perturbed[0][1], skip_special_tokens=True, clean_up_tokenization_spaces=True) + '\n')


    all_preds = []
    all_preds_perturbed = []
    all_golds = []
    all_pred_probs = None
    all_pred_probs_perturbed = None

    eval_dataloader = list(eval_dataloader)
    eval_dataloader_perturbed = list(eval_dataloader_perturbed)

    if training_args.temperature_scaling > 1.0:
        print('Will be performing temperature_scaling with temperature ' + str(training_args.temperature_scaling))

    for i in range(len(eval_dataloader)):
        batch = eval_dataloader[i]
        batch_perturbed = eval_dataloader_perturbed[i]

        preds_batch, gold_batch, logits = forward_pass(model, batch , args, return_logits=True)
        preds_batch_perturbed, _, logits_perturbed = forward_pass(model, batch_perturbed , args, return_logits=True)

        if training_args.temperature_scaling > 1.0:
            # Scale Logits perturbed with temperature scaling
            logits_perturbed = do_temperature_scaling(logits_perturbed, training_args.temperature_scaling)
            logits = do_temperature_scaling(logits, training_args.temperature_scaling)



        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        logits_perturbed = torch.nn.functional.log_softmax(logits_perturbed, dim=-1)

        if all_pred_probs is None:
            all_pred_probs = logits.data.cpu().numpy()
        else:
            all_pred_probs = np.concatenate([all_pred_probs, logits.data.cpu().numpy()])

        if all_pred_probs_perturbed is None:
            all_pred_probs_perturbed = logits_perturbed.data.cpu().numpy()
        else:
            all_pred_probs_perturbed = np.concatenate([all_pred_probs_perturbed, logits_perturbed.data.cpu().numpy()])

        all_preds.extend(preds_batch)
        all_preds_perturbed.extend(preds_batch_perturbed)
        all_golds.extend(gold_batch)

    fp_log.write('Completed Forward Pass on all examples ...' + '\n')
    fp_log.write('Going to calculate statistics' + '\n')

    get_statistics(all_preds, all_golds, all_preds_perturbed, all_pred_probs,
                   all_pred_probs_perturbed, fp_log, fp_preds_perturbed)

    print('Done with statistics. Returning')
    return

data_args.eval_batch_size = training_args.eval_batch_size
for k , filename in generated_filenames.items():
    print('Evaluating Perturbation ' + str(filename))
    if filename is None:
        continue
    evaluate_one_perturbation(data_args, training_args, filename, training_args.save_perturb_output_dir )


