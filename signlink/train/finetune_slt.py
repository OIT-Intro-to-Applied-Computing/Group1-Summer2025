import argparse, yaml, torch, os, csv, datetime # Added csv, datetime
from torch.utils.data import DataLoader
from ..datamodules.how2sign_rgb import How2SignRGB
from ..datamodules.collate import collate_fn
from ..tokenization.spm import SentencePieceTokenizer
from ..models.sign2text_model import Sign2TextModel
from ..utils.distributed import seed_all, device
from ..utils.metrics import bleu, chrf # Imported bleu, chrf
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    seed_all(42)
    dev = device()

    tok = SentencePieceTokenizer(cfg["tokenizer"]["spm_model"])
    train_set = How2SignRGB(cfg["data"]["train_manifest"], img_size=cfg["data"]["img_size"], frames=cfg["data"]["frames"], frame_drop_rate=cfg["data"].get("frame_drop_rate", 0.0)) # Pass frame_drop_rate
    val_set   = How2SignRGB(cfg["data"]["val_manifest"],   img_size=cfg["data"]["img_size"], frames=cfg["data"]["frames"], frame_drop_rate=0.0) # No frame drop for validation

    coll = lambda b: collate_fn(b, tok)
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], collate_fn=coll)
    val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"], collate_fn=coll) # Set val_loader batch_size to 1

    vocab_size = tok.sp.get_piece_size()
    model = Sign2TextModel(vocab_size=vocab_size, d_model=cfg["model"]["d_model"], decoder_layers=cfg["decoder"]["num_layers"], nhead=cfg["decoder"]["nhead"], use_pose=cfg["model"]["use_pose"], enc_layers=cfg["model"]["encoder_layers"]).to(dev)

    # Explicitly cast lr to float
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=cfg["train"]["wd"])

    # Add LR scheduler and early stopping
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.2, patience=10) # Changed factor to 0.2
    best_val = float("inf"); no_improve = 0; EARLY_STOP = 15 # Increased EARLY_STOP

    os.makedirs("checkpoints", exist_ok=True)
    
    # Setup logging to file
    log_file_path = "training_log.csv"
    log_header = ["timestamp", "epoch", "train_loss", "val_loss", "val_bleu", "val_chrf"]
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_header)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        total_train_loss = 0.0 # Changed from total
        num_train_batches = 0
        for batch in pbar:
            pix = batch["pixel_values"].to(dev)
            tgt = batch["tgt_ids"].to(dev)
            pose = batch["pose"].to(dev) if batch["pose"] is not None else None # Extract pose
            loss = model(pix, tgt, pose=pose)  # Pass pose to model
            optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
            total_train_loss += loss.item()
            num_train_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = total_train_loss / max(1, num_train_batches) # Calculate average train loss

        val_loss, gts, preds = evaluate_and_collect(model, val_loader, dev, tok, cfg["inference"]["max_len"]) # Modified call
        val_bleu = bleu(gts, preds)
        val_chrf = chrf(gts, preds)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:.2f} | Val chrF: {val_chrf:.2f}")
        
        # Log to file
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch + 1, avg_train_loss, val_loss, val_bleu, val_chrf])

        # Early stopping and LR scheduling
        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss; no_improve = 0
            torch.save(model.state_dict(), "checkpoints/slt_best.pth")
            print("Saved checkpoints/slt_best.pth")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print("Early stopping: no improvement.")
                break

# Modified evaluate function to return gts and preds
def evaluate_and_collect(model, loader, dev, tok, max_len):
    model.eval(); tot=0.0; n=0
    gts, preds = [], [] # Initialize lists for ground truths and predictions
    with torch.no_grad():
        for batch in loader:
            pix = batch["pixel_values"].to(dev)
            tgt = batch["tgt_ids"].to(dev)
            pose = batch["pose"].to(dev) if batch["pose"] is not None else None
            
            loss = model(pix, tgt, pose=pose)
            tot += loss.item(); n+=1

            # Generate predictions for metrics
            ids = model.generate(pix, pose=pose, max_len=max_len)
            pred = tok.decode(ids)
            gt = batch["tgt_ids"][0].tolist()
            gt_text = tok.decode(gt)
            
            gts.append(gt_text)
            preds.append(pred)
            
    return tot/max(n,1), gts, preds # Return loss, gts, and preds

if __name__ == "__main__":
    main()
