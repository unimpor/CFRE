import argparse
import glob
import json
import os
import re
import string
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

class RewardMetrics:
    def __init__(self, metrics_name) -> None:
        metrics_all = {
            "F1": self.F1,
            "recall": self.recall,
            "precision": self.precision
        }
        self.metrics_name = metrics_name
        self.metrics_func = metrics_all[metrics_name]
    
    def postprocess_pred(self, prediction):
        res = [p for p in prediction.split("\n") if 'ans:' in p and 'none' not in p.lower()]
        if len(res) >= 1:
            res = [p for p in res if "ans: not available" not in p.lower() and "ans: no information available" not in p.lower()]
        return sorted(remove_duplicates(res), key=len, reverse=True)
    
    def postprocess_ans(self, answer, question):
        answer = sorted(remove_duplicates(answer), key=len, reverse=True)
        if 'when' in question.lower() or 'what year' in question.lower():
            for idx in range(len(answer)):
                if '-' in answer[idx] and answer[idx].split('-')[0].isdigit():
                    answer[idx] = answer[idx].split('-')[0]
        return answer
    
    def calc_r(self, prediction, answer, question):
        prediction, answer = deepcopy(prediction), deepcopy(answer)
        prediction = self.postprocess_pred(prediction)
        answer = self.postprocess_ans(answer, question)
        num_pred = len(prediction)
        num_ans = len(answer)
        double_check = any([keyword in question.lower() for keyword in ['when', 'what year', 'which year', 'where', 'sport', "what countr", "language", 'nba finals', 'world series']])
        matched = 0.
        for a in answer:
            for pred in prediction:
                if match(pred, a):
                    matched += 1
                    prediction.remove(pred)
                    break
                elif double_check:
                    if match(a, pred.split('ans:')[-1].strip()) or match(a, pred):
                        matched += 1
                        prediction.remove(pred)
                        break
        return self.metrics_func(matched, num_pred, num_ans)
    
    def precision(self, matched, num_pred, num_ans):
        if num_pred == 0:
            return 0, 0, 0
        return matched / num_pred
    
    def recall(self, matched, num_pred, num_ans):
        return matched / num_ans
    
    def F1(self, matched, num_pred, num_ans):
        precision = self.precision(matched, num_pred, num_ans)
        recall = self.recall(matched, num_pred, num_ans)
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)
