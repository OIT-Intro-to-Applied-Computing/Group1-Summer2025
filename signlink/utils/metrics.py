import sacrebleu
from typing import List

def bleu(refs: List[str], hyps: List[str]) -> float:
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)

def chrf(refs: List[str], hyps: List[str]) -> float:
    return float(sacrebleu.corpus_chrf(hyps, [refs]).score)
