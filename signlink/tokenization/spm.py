import sentencepiece as spm

class SentencePieceTokenizer:
    def __init__(self, model_path, bos_id=1, eos_id=2):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.bos_id = bos_id
        self.eos_id = eos_id
    def encode(self, text):
        ids = self.sp.encode(text, out_type=int)
        return [self.bos_id] + ids + [self.eos_id]
    def decode(self, ids):
        # strip at EOS if present
        out = []
        for i in ids:
            if i == self.eos_id: break
            if i in (self.bos_id, 0): continue
            out.append(i)
        return self.sp.decode(out)
