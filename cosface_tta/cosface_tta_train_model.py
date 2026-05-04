import os
import torch
import wandb
from tqdm import tqdm
from cosface_tta_eval import eval_tta
from torch.autograd import Variable
from utils.mixup_utils import mixup_data, mixup_criterion

def train(model,
          classifier,
          device,
          trainloader,
          valloader,
          testloader,
          metric_loss,
          miner,
          criterion,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          best_val_acc):
    best_acc = best_val_acc
    for epoch in range(start_epoch + 1, end_epoch + 1):
        log_path = os.path.join(save_path, 'log.txt')
        model.train()
        classifier.train()
        print('Training %d epoch' % epoch)

        lr = next(iter(optimizer.param_groups))['lr']
        turn = True
        epoch_train_loss = 0.0
        for _, data in enumerate(tqdm(trainloader)):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if turn:
                p, embedding = model(images)
                pminer = miner(p, labels)
                p_mloss = metric_loss(p, labels, pminer)
                # Pass labels to classifier for CosFace margin
                logits = classifier(embedding, labels=labels)
                ce_loss = criterion(logits, labels)
                total_loss = ce_loss + p_mloss
            else:
                images, labels_a, labels_b, lam = mixup_data(images, labels, 1.0, True)
                images, labels_a, labels_b = map(Variable, (images, labels_a, labels_b))
                p, embedding = model(images)
                # For mixup, we don't apply the margin to a single label because there are two labels.
                # Just pass None for labels to get the raw cosine similarities for both
                logits = classifier(embedding, labels=None)
                ce_loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                total_loss = ce_loss
            turn = not turn

            total_loss.backward()
            optimizer.step()
            epoch_train_loss += total_loss.item()
            
        scheduler.step()
        
        with open(log_path, 'a') as f:
            f.write('\nEPOCH' + str(epoch) + '\n')
            # eval valset
            loss_avg, metric_loss_avg, accuracy = eval_tta(model, classifier, device, valloader, metric_loss, miner, criterion, 'Validation')
            print('Validation set: Avg Val CE Loss: {:.4f}; Avg Val Metric Loss: {:.4f}; Val accuracy: {:.2f}% '.format(
                loss_avg, metric_loss_avg, 100. * accuracy))
            f.write('Validation set: Avg Val CE Loss: {:.4f}; Avg Val Metric Loss: {:.4f}; Val accuracy: {:.2f}% \n'.format(
                loss_avg, metric_loss_avg, 100. * accuracy))
            
            # eval testset
            test_loss_avg, test_metric_loss_avg, test_accuracy = eval_tta(model, classifier, device, testloader, metric_loss, miner, criterion, 'Test')
            print('Test set: Avg Test CE Loss: {:.4f}; Avg Test Metric Loss: {:.4f}; Test accuracy: {:.2f}%'.format(
                test_loss_avg, test_metric_loss_avg, 100. * test_accuracy))
            f.write('Test set: Avg Test CE Loss: {:.4f}; Avg Test Metric Loss: {:.4f}; Test accuracy: {:.2f}%\n'.format(
                test_loss_avg, test_metric_loss_avg, 100. * test_accuracy))
                
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss / len(trainloader),
            "val_ce_loss": loss_avg,
            "val_metric_loss": metric_loss_avg,
            "val_acc": accuracy,
            "test_ce_loss": test_loss_avg,
            "test_metric_loss": test_metric_loss_avg,
            "test_acc": test_accuracy,
            "learning_rate": lr
        })
                
        if accuracy > best_acc:
            print('Saving best model!')
            with open(log_path, 'a') as f:
                f.write('Saving best model!\n')
            best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': best_acc,
            }, os.path.join(save_path, 'best_model.pth'))
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': best_acc,
        }, os.path.join(save_path, 'current_model.pth'))
