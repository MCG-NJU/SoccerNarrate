import json
import argparse
from collections import defaultdict


def is_correct(predict, answer):
    # Upstream implementation note.
    if len(answer) == 1:
        return answer[0] == predict[0]
    elif len(answer) != 1 and answer[0] in ['A', 'B', 'C', 'D']:
        return answer[0] == predict[0]
    elif len(answer) != 1 and answer[0] not in ['A', 'B', 'C', 'D']:
        return predict[4:].lower() in answer.lower()
