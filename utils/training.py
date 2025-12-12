import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from utils.utils import accuracy, AverageMeter
from utils.model_factory import create_distillation_model, print_model_parameters
from utils.checkpoint import save_checkpoint


class Trainer:
    """
    Unified trainer for all knowledge distillation methods.
    Handles interleaved adversarial training for DisDKD.
    """

    def __init__(self, teacher, student, num_classes, criterion, loss_tracker, device, args):
        self.teacher = teacher
        self.student = student
        self.num_classes = num_classes
        self.criterion = criterion
        self.loss_tracker = loss_tracker
        self.device = device
        self.args = args
        self.method = args.method

        # Build distillation model
        if self.method != "Pretraining":
            self.distill_model = self._build_distillation_model()
        else:
            self.distill_model = None

        # Setup optimizers
        self._setup_optimizers()

        # Track best accuracy
        self.best_val_acc = 0.0

    def _build_distillation_model(self):
        """Build the distillation model based on method."""
        from utils.model_factory import create_distillation_model

        return create_distillation_model(
            self.args,
            self.teacher,
            self.student,
            self.num_classes
        ).to(self.device)

    def _setup_optimizers(self):
        """Setup optimizers based on method."""
        if self.method == "Pretraining":
            # Only train teacher
            self.optimizer = self._get_optimizer(self.teacher.parameters())
            self.scheduler = self._get_scheduler(self.optimizer)
        elif self.method == "DisDKD":
            # For DisDKD, we create optimizers per phase
            self.optimizer_D = self.distill_model.get_discriminator_optimizer(
                lr=self.args.disdkd_phase1_lr,
                weight_decay=self.args.weight_decay
            )
            self.optimizer_G = self.distill_model.get_generator_optimizer(
                lr=self.args.disdkd_phase2_lr,
                weight_decay=self.args.weight_decay
            )
            # Phase 2: DKD optimizer (created later when transitioning)
            self.optimizer_DKD = None
            self.scheduler = None  # Manage LR manually for DisDKD
        else:
            # Standard methods
            self.optimizer = self._get_optimizer(self.student.parameters())
            self.scheduler = self._get_scheduler(self.optimizer)

    def _get_optimizer(self, params):
        """Get optimizer based on args."""
        if self.args.optimizer == "adam":
            return torch.optim.Adam(
                params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")

    def _get_scheduler(self, optimizer):
        """Get learning rate scheduler."""
        if self.args.step_size > 0:
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.args.step_size,
                gamma=self.args.lr_decay
            )
        return None

    def train(self, train_loader, val_loader):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training with method: {self.method}")
        print(f"{'='*60}\n")

        if self.method == "DisDKD":
            return self._train_disdkd(train_loader, val_loader)
        else:
            return self._train_standard(train_loader, val_loader)

    def _train_disdkd(self, train_loader, val_loader):
        """
        Training loop for DisDKD with interleaved adversarial training.

        Phase 1 (first N epochs): Interleaved D/G training
        Phase 2 (remaining epochs): DKD fine-tuning
        """
        phase1_epochs = self.args.disdkd_phase1_epochs
        total_epochs = self.args.epochs
        k_disc_steps = getattr(self.args, 'disdkd_k_disc_steps', 1)

        # Phase 1: Adversarial (interleaved D/G)
        print(f"\n{'='*60}")
        print(f"PHASE 1: Adversarial Training (Interleaved D/G)")
        print(f"  Epochs: {phase1_epochs}")
        print(f"  Discriminator steps per batch: {k_disc_steps}")
        print(f"  Generator steps per batch: 1")
        print(f"  Discriminator LR: {self.args.disdkd_phase1_lr}")
        print(f"  Generator LR: {self.args.disdkd_phase2_lr}")
        print(f"{'='*60}\n")

        self.distill_model.set_phase(1)

        for epoch in range(1, phase1_epochs + 1):
            train_metrics = self._train_epoch_phase1(train_loader, epoch, k_disc_steps)
            val_metrics = self._validate_epoch_phase1(val_loader, epoch)

            # Log metrics
            self.loss_tracker.log_epoch(
                epoch=epoch,
                phase="train",
                losses={
                    "disdkd_phase": 1,
                    "disc_loss": train_metrics["disc_loss"],
                    "disc_acc": train_metrics["disc_acc"],
                    "gen_loss": train_metrics["gen_loss"],
                    "fool_rate": train_metrics["fool_rate"],
                    "teacher_pred_mean": train_metrics["teacher_pred_mean"],
                    "student_pred_mean": train_metrics["student_pred_mean"],
                    "total": train_metrics["disc_loss"] + train_metrics["gen_loss"],
                },
                accuracy=0.0
            )

            self.loss_tracker.log_epoch(
                epoch=epoch,
                phase="val",
                losses={
                    "disdkd_phase": 1,
                    "disc_loss": val_metrics["disc_loss"],
                    "disc_acc": val_metrics["disc_acc"],
                    "gen_loss": val_metrics["gen_loss"],
                    "fool_rate": val_metrics["fool_rate"],
                    "teacher_pred_mean": val_metrics["teacher_pred_mean"],
                    "student_pred_mean": val_metrics["student_pred_mean"],
                    "total": val_metrics["disc_loss"] + val_metrics["gen_loss"],
                },
                accuracy=0.0
            )

            print(f"\nEpoch {epoch}/{phase1_epochs} [Phase 1] Summary:")
            print(f"  Train - D_loss: {train_metrics['disc_loss']:.4f}, "
                  f"D_acc: {train_metrics['disc_acc']:.2f}%, "
                  f"G_loss: {train_metrics['gen_loss']:.4f}, "
                  f"Fool: {train_metrics['fool_rate']:.2f}%")
            print(f"  Val   - D_loss: {val_metrics['disc_loss']:.4f}, "
                  f"D_acc: {val_metrics['disc_acc']:.2f}%, "
                  f"G_loss: {val_metrics['gen_loss']:.4f}, "
                  f"Fool: {val_metrics['fool_rate']:.2f}%")
            print("-" * 60)

        # Transition to Phase 2
        print(f"\n{'='*60}")
        print(f"PHASE 2: DKD Fine-tuning")
        print(f"  Epochs: {total_epochs - phase1_epochs}")
        print(f"  Learning rate: {self.args.disdkd_phase3_lr}")
        print(f"{'='*60}\n")

        self.distill_model.set_phase(2)
        self.distill_model.discard_adversarial_components()

        # Create DKD optimizer
        self.optimizer_DKD = self.distill_model.get_dkd_optimizer(
            lr=self.args.disdkd_phase3_lr,
            weight_decay=self.args.weight_decay
        )
        
        # Create scheduler for Phase 2
        if self.args.step_size > 0:
            self.scheduler_DKD = torch.optim.lr_scheduler.StepLR(
                self.optimizer_DKD,
                step_size=self.args.step_size,
                gamma=self.args.lr_decay
            )
        else:
            self.scheduler_DKD = None

        # Phase 2: DKD
        for epoch in range(phase1_epochs + 1, total_epochs + 1):
            train_metrics = self._train_epoch_phase2(train_loader, epoch)
            val_metrics = self._validate_epoch_phase2(val_loader, epoch)

            # Log metrics
            self.loss_tracker.log_epoch(
                epoch=epoch,
                phase="train",
                losses={
                    "disdkd_phase": 2,
                    "ce": train_metrics["ce_loss"],
                    "dkd": train_metrics["dkd_loss"],
                    "total": train_metrics["total_loss"],
                },
                accuracy=train_metrics["accuracy"]
            )

            self.loss_tracker.log_epoch(
                epoch=epoch,
                phase="val",
                losses={
                    "disdkd_phase": 2,
                    "ce": val_metrics["ce_loss"],
                    "dkd": val_metrics["dkd_loss"],
                    "total": val_metrics["total_loss"],
                },
                accuracy=val_metrics["accuracy"]
            )

            # Step scheduler
            if self.scheduler_DKD:
                self.scheduler_DKD.step()

            # Track best model
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self._save_checkpoint(epoch, is_best=True)
                print(f"  ⭐ New best val accuracy: {self.best_val_acc:.2f}%")

            print(f"\nEpoch {epoch}/{total_epochs} [Phase 2] Summary:")
            print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, "
                  f"CE: {train_metrics['ce_loss']:.4f}, "
                  f"DKD: {train_metrics['dkd_loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, "
                  f"CE: {val_metrics['ce_loss']:.4f}, "
                  f"DKD: {val_metrics['dkd_loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.2f}%")
            print("-" * 60)

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")

        return self.best_val_acc

    def _train_epoch_phase1(self, train_loader, epoch, k_disc_steps):
        """Train one epoch of Phase 1 (interleaved D/G) with tqdm progress."""
        self.distill_model.train()

        # Accumulators for batch-level metrics
        disc_loss_sum = 0.0
        disc_acc_sum = 0.0
        teacher_pred_sum = 0.0
        gen_loss_sum = 0.0
        fool_rate_sum = 0.0
        student_pred_sum = 0.0
        
        num_batches = 0
        num_disc_updates = 0

        # Progress bar
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch} [Phase 1] Train",
            ncols=120,
            leave=False
        )

        for batch_idx, (inputs, _) in enumerate(pbar):
            inputs = inputs.to(self.device)

            # ===== Train Discriminator for k steps =====
            batch_disc_loss = 0.0
            batch_disc_acc = 0.0
            batch_teacher_pred = 0.0
            
            for disc_step in range(k_disc_steps):
                self.optimizer_D.zero_grad()

                outputs = self.distill_model(inputs, mode='discriminator')
                loss_D = outputs['disc_loss']
                loss_D.backward()
                self.optimizer_D.step()

                batch_disc_loss += outputs["disc_loss"].item()
                batch_disc_acc += outputs["disc_accuracy"]
                batch_teacher_pred += outputs["teacher_pred_mean"]
                num_disc_updates += 1

            # Average discriminator metrics for this batch
            batch_disc_loss /= k_disc_steps
            batch_disc_acc /= k_disc_steps
            batch_teacher_pred /= k_disc_steps

            # ===== Train Generator for 1 step =====
            self.optimizer_G.zero_grad()

            outputs = self.distill_model(inputs, mode='generator')
            loss_G = outputs['gen_loss']
            loss_G.backward()
            self.optimizer_G.step()

            batch_gen_loss = outputs["gen_loss"].item()
            batch_fool_rate = outputs["fool_rate"]
            batch_student_pred = outputs["student_pred_mean"]

            # Accumulate
            disc_loss_sum += batch_disc_loss
            disc_acc_sum += batch_disc_acc
            teacher_pred_sum += batch_teacher_pred
            gen_loss_sum += batch_gen_loss
            fool_rate_sum += batch_fool_rate
            student_pred_sum += batch_student_pred
            num_batches += 1

            # Update progress bar with running averages
            pbar.set_postfix({
                'D_loss': f'{disc_loss_sum/num_batches:.4f}',
                'D_acc': f'{100*disc_acc_sum/num_batches:.1f}%',
                'G_loss': f'{gen_loss_sum/num_batches:.4f}',
                'Fool': f'{100*fool_rate_sum/num_batches:.1f}%',
            })

        pbar.close()

        # Return epoch averages
        return {
            "disc_loss": disc_loss_sum / num_batches,
            "disc_acc": 100 * disc_acc_sum / num_batches,
            "gen_loss": gen_loss_sum / num_batches,
            "fool_rate": 100 * fool_rate_sum / num_batches,
            "teacher_pred_mean": teacher_pred_sum / num_batches,
            "student_pred_mean": student_pred_sum / num_batches,
        }

    def _validate_epoch_phase1(self, val_loader, epoch):
        """Validate one epoch of Phase 1 with tqdm progress."""
        self.distill_model.eval()

        disc_loss_sum = 0.0
        disc_acc_sum = 0.0
        teacher_pred_sum = 0.0
        gen_loss_sum = 0.0
        fool_rate_sum = 0.0
        student_pred_sum = 0.0
        num_batches = 0

        pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch} [Phase 1] Val",
            ncols=120,
            leave=False
        )

        with torch.no_grad():
            for inputs, _ in pbar:
                inputs = inputs.to(self.device)

                # Evaluate discriminator
                disc_outputs = self.distill_model(inputs, mode='discriminator')

                # Evaluate generator
                gen_outputs = self.distill_model(inputs, mode='generator')

                disc_loss_sum += disc_outputs["disc_loss"].item()
                disc_acc_sum += disc_outputs["disc_accuracy"]
                teacher_pred_sum += disc_outputs["teacher_pred_mean"]
                gen_loss_sum += gen_outputs["gen_loss"].item()
                fool_rate_sum += gen_outputs["fool_rate"]
                student_pred_sum += gen_outputs["student_pred_mean"]
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f'{disc_loss_sum/num_batches:.4f}',
                    'D_acc': f'{100*disc_acc_sum/num_batches:.1f}%',
                    'G_loss': f'{gen_loss_sum/num_batches:.4f}',
                    'Fool': f'{100*fool_rate_sum/num_batches:.1f}%',
                })

        pbar.close()

        return {
            "disc_loss": disc_loss_sum / num_batches,
            "disc_acc": 100 * disc_acc_sum / num_batches,
            "gen_loss": gen_loss_sum / num_batches,
            "fool_rate": 100 * fool_rate_sum / num_batches,
            "teacher_pred_mean": teacher_pred_sum / num_batches,
            "student_pred_mean": student_pred_sum / num_batches,
        }

    def _train_epoch_phase2(self, train_loader, epoch):
        """Train one epoch of Phase 2 (DKD) with tqdm progress."""
        self.distill_model.train()

        ce_loss_sum = 0.0
        dkd_loss_sum = 0.0
        total_loss_sum = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch} [Phase 2] Train",
            ncols=120,
            leave=False
        )

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer_DKD.zero_grad()

            outputs = self.distill_model(inputs, targets=targets)
            student_logits = outputs["student_logits"]
            dkd_loss = outputs["dkd_loss"]
            ce_loss = self.criterion(student_logits, targets)

            # Total loss: CE + DKD
            total_loss = self.args.alpha * ce_loss + self.args.beta * dkd_loss
            total_loss.backward()
            self.optimizer_DKD.step()

            # Compute accuracy
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            ce_loss_sum += ce_loss.item()
            dkd_loss_sum += dkd_loss.item()
            total_loss_sum += total_loss.item()

            # Update progress bar with running averages
            num_batches = pbar.n + 1
            pbar.set_postfix({
                'loss': f'{total_loss_sum/num_batches:.4f}',
                'ce': f'{ce_loss_sum/num_batches:.4f}',
                'dkd': f'{dkd_loss_sum/num_batches:.4f}',
                'acc': f'{100*correct/total:.2f}%',
            })

        pbar.close()

        num_batches = len(train_loader)
        return {
            "ce_loss": ce_loss_sum / num_batches,
            "dkd_loss": dkd_loss_sum / num_batches,
            "total_loss": total_loss_sum / num_batches,
            "accuracy": 100.0 * correct / total,
        }

    def _validate_epoch_phase2(self, val_loader, epoch):
        """Validate one epoch of Phase 2 with tqdm progress."""
        self.distill_model.eval()

        ce_loss_sum = 0.0
        dkd_loss_sum = 0.0
        total_loss_sum = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch} [Phase 2] Val",
            ncols=120,
            leave=False
        )

        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.distill_model(inputs, targets=targets)
                student_logits = outputs["student_logits"]
                dkd_loss = outputs["dkd_loss"]
                ce_loss = self.criterion(student_logits, targets)

                total_loss = self.args.alpha * ce_loss + self.args.beta * dkd_loss

                # Compute accuracy
                _, predicted = student_logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                ce_loss_sum += ce_loss.item()
                dkd_loss_sum += dkd_loss.item()
                total_loss_sum += total_loss.item()

                # Update progress bar
                num_batches = pbar.n + 1
                pbar.set_postfix({
                    'loss': f'{total_loss_sum/num_batches:.4f}',
                    'ce': f'{ce_loss_sum/num_batches:.4f}',
                    'dkd': f'{dkd_loss_sum/num_batches:.4f}',
                    'acc': f'{100*correct/total:.2f}%',
                })

        pbar.close()

        num_batches = len(val_loader)
        return {
            "ce_loss": ce_loss_sum / num_batches,
            "dkd_loss": dkd_loss_sum / num_batches,
            "total_loss": total_loss_sum / num_batches,
            "accuracy": 100.0 * correct / total,
        }

    def _train_standard(self, train_loader, val_loader):
        """Standard training loop for other methods."""
        raise NotImplementedError(
            "Standard training not yet implemented. Use DisDKD for now."
        )

    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        save_checkpoint(
            self.distill_model.student if hasattr(self.distill_model, 'student') else self.student,
            self.optimizer_DKD if hasattr(self, 'optimizer_DKD') else self.optimizer,
            epoch,
            self.best_val_acc,
            self.args,
            is_best=is_best
        )
