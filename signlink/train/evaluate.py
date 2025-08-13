import argparse, torch, yaml, pandas as pd, os
from torch.utils.data import DataLoader
from ..datamodules.how2sign_rgb import How2SignRGB
from ..datamodules.collate import collate_fn
from ..tokenization.spm import SentencePieceTokenizer
from ..models.sign2text_model import Sign2TextModel
from ..utils.distributed import device
from ..utils.metrics import bleu, chrf
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/slt_temporal1d_base.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    tok = SentencePieceTokenizer(cfg["tokenizer"]["spm_model"])
    ds = How2SignRGB(args.manifest, img_size=cfg["data"]["img_size"], frames=cfg["data"]["frames"])
    coll = lambda b: collate_fn(b, tok)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=coll)
    dev = device()
    vocab_size = tok.sp.get_piece_size()
    model = Sign2TextModel(vocab_size=vocab_size, d_model=cfg["model"]["d_model"], decoder_layers=cfg["decoder"]["num_layers"], nhead=cfg["decoder"]["nhead"], use_pose=cfg["model"]["use_pose"], enc_layers=cfg["model"]["encoder_layers"]).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev), strict=False) # Allow partial loading
    model.eval()

    names, gts, preds = [], [], []
    with torch.no_grad():
        for batch in tqdm(dl):
            pix = batch["pixel_values"].to(dev)
            pose = batch["pose"].to(dev) if batch["pose"] is not None else None # Extract pose
            ids = model.generate(pix, pose=pose, max_len=cfg["inference"]["max_len"]) # Pass pose to model
            pred = tok.decode(ids)
            gt = batch["tgt_ids"][0].tolist()
            gt_text = tok.decode(gt)
            names.append(batch["names"][0]); gts.append(gt_text); preds.append(pred)

    # length stats
    lengths = [len(p.split()) for p in preds]
    print("Pred len avg:", sum(lengths)/len(lengths),
          "median:", sorted(lengths)[len(lengths)//2],
          "zero-len %:", sum(l==0 for l in lengths)/len(lengths))

    b = bleu(gts, preds); c = chrf(gts, preds)
    print(f"BLEU: {b:.2f}  chrF: {c:.2f}")
    
    # case-insensitive trend (in addition to strict)
    refs_lc = [r.lower() for r in gts]
    preds_lc = [p.lower() for p in preds]
    b_ci = bleu(refs_lc, preds_lc); c_ci = chrf(refs_lc, preds_lc)
    print(f"(case-insensitive) BLEU: {b_ci:.2f}  chrF: {c_ci:.2f}")

    out = pd.DataFrame({"SENTENCE_NAME": names, "GT": gts, "PRED": preds})
    os.makedirs("outputs", exist_ok=True)
    out.to_csv("outputs/preds.csv", index=False)
    print("Saved outputs/preds.csv")

if __name__ == "__main__":
    main()
