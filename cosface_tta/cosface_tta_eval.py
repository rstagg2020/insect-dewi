import torch
from tqdm import tqdm

def eval_tta(model, classifier, device, loader, metric_loss, miner, criterion, split):
    model.eval()
    classifier.eval()
    print('Evaluating model with TTA on ' + split + ' data')

    ce_loss_sum = 0
    metric_loss_sum = 0
    correct = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            # Original Image
            p_orig, emb_orig = model(images)
            logits_orig = classifier(emb_orig)
            
            # Flipped Image
            images_flipped = torch.flip(images, dims=[3])
            p_flip, emb_flip = model(images_flipped)
            logits_flip = classifier(emb_flip)
            
            # Average embeddings or logits
            p = (p_orig + p_flip) / 2.0
            logits = (logits_orig + logits_flip) / 2.0

            pminer = miner(p, labels)
            p_mloss = metric_loss(p, labels, pminer)
            ce_loss = criterion(logits, labels)

            ce_loss_sum += ce_loss.item()
            metric_loss_sum += p_mloss.item()

            pred = logits.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    loss_avg = ce_loss_sum / (i+1)
    metric_loss_avg = metric_loss_sum / (i+1)

    accuracy = correct / len(loader.dataset)

    return loss_avg, metric_loss_avg, accuracy
