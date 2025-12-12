import torch
import torch.nn as nn
import torchvision.models as models


class TeacherModel(nn.Module):
    """Teacher model wrapper."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 100,
        weights_path: str = None,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model = self._build_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
        if weights_path:
            self.load_teacher_weights(weights_path)

    def _build_model(self, model_name, pretrained, num_classes):
        """Build the model architecture."""
        if model_name.startswith("resnet"):
            # Extract number from model name (e.g., "50" from "resnet50")
            num = model_name[len("resnet") :]
            if pretrained:
                weights_enum = getattr(models, f"ResNet{num}_Weights")
                weights = weights_enum.DEFAULT
                print(f"[Teacher] Loading pretrained ImageNet weights for {model_name}")
            else:
                weights = None
                print(f"[Teacher] Initializing {model_name} with random weights")

            model = getattr(models, model_name)(weights=weights)

            # Replace final head if needed
            if num_classes != model.fc.out_features:
                model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif model_name.startswith("vgg"):
            if pretrained:
                weights_enum = getattr(models, f"{model_name.upper()}_Weights")
                weights = weights_enum.DEFAULT
                print(f"[Teacher] Loading pretrained ImageNet weights for {model_name}")
            else:
                weights = None
                print(f"[Teacher] Initializing {model_name} with random weights")

            model = getattr(models, model_name)(weights=weights)

            # Replace final head if needed
            if num_classes != model.classifier[-1].out_features:
                layers = list(model.classifier)
                for i in reversed(range(len(layers))):
                    if isinstance(layers[i], nn.Linear):
                        in_feats = layers[i].in_features
                        layers[i] = nn.Linear(in_feats, num_classes)
                        break
                model.classifier = nn.Sequential(*layers)

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model

    def load_teacher_weights(self, weights_path):
        """Load pretrained weights for teacher."""
        sd = torch.load(weights_path, weights_only=False)
        self.model.load_state_dict(sd, strict=False)
        print(f"[Teacher] Loaded custom weights from {weights_path}")

    def forward(self, x):
        return self.model(x)


class StudentModel(nn.Module):
    """Student model wrapper."""

    def __init__(
        self, model_name: str, num_classes: int = 100, weights_path: str = None
    ):
        super().__init__()
        self.model = self._build_model(
            model_name, pretrained=False, num_classes=num_classes
        )
        if weights_path:
            self.load_student_weights(weights_path)

    def _build_model(self, model_name, pretrained, num_classes):
        """Build the model architecture."""
        if model_name.startswith("resnet"):
            model = getattr(models, model_name)(weights=None)
            if num_classes != model.fc.out_features:
                model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif model_name.startswith("vgg"):
            model = getattr(models, model_name)(weights=None)
            if num_classes != model.classifier[-1].out_features:
                layers = list(model.classifier)
                for i in reversed(range(len(layers))):
                    if isinstance(layers[i], nn.Linear):
                        in_feats = layers[i].in_features
                        layers[i] = nn.Linear(in_feats, num_classes)
                        break
                model.classifier = nn.Sequential(*layers)

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model

    def load_student_weights(self, weights_path):
        """Load pretrained weights for student."""
        sd = torch.load(weights_path, weights_only=False)
        self.model.load_state_dict(sd, strict=False)
        print(f"[Student] Loaded weights from {weights_path}")

    def forward(self, x):
        return self.model(x)


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
            self.teacher,
            self.student,
            self.method,
            self.args
        ).to(self.device)

    def _setup_optimizers(self):
        """Setup optimizers based on method."""
        if self.method == "Pretraining":
            # Only train teacher
            self.optimizer = self._get_optimizer(self.teacher.parameters())
            self.scheduler = self._get_scheduler(self.optimizer)
        elif self.method == "DisDKD":
            # For DisDKD, we'll create optimizers per phase
            # Phase 1: discriminator and generator optimizers
            self.optimizer_D = self.distill_model.get_discriminator_optimizer(
                lr=self.args.disdkd_phase1_lr,
                weight_decay=self.args.weight_decay
            )
            self.optimizer_G = self.distill_model.get_generator_optimizer(
                lr=self.args.disdkd_phase2_lr,  # generator LR
                weight_decay=self.args.weight_decay
            )
            # Phase 2: DKD optimizer (created later when transitioning to Phase 2)
            self.optimizer_DKD = None
            self.scheduler = None  # We'll manage LR manually for DisDKD
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
        k_disc_steps = self.args.disdkd_k_disc_steps if hasattr(self.args, 'disdkd_k_disc_steps') else 1

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
                },
                accuracy=0.0  # No classification accuracy in Phase 1
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
                },
                accuracy=0.0
            )

            print(f"Epoch {epoch}/{phase1_epochs} [Phase 1] - "
                  f"D_loss: {train_metrics['disc_loss']:.4f}, "
                  f"D_acc: {train_metrics['disc_acc']:.2f}%, "
                  f"G_loss: {train_metrics['gen_loss']:.4f}, "
                  f"Fool_rate: {train_metrics['fool_rate']:.2f}%")

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
                    "ce_loss": train_metrics["ce_loss"],
                    "dkd_loss": train_metrics["dkd_loss"],
                    "total_loss": train_metrics["total_loss"],
                },
                accuracy=train_metrics["accuracy"]
            )

            self.loss_tracker.log_epoch(
                epoch=epoch,
                phase="val",
                losses={
                    "disdkd_phase": 2,
                    "ce_loss": val_metrics["ce_loss"],
                    "dkd_loss": val_metrics["dkd_loss"],
                    "total_loss": val_metrics["total_loss"],
                },
                accuracy=val_metrics["accuracy"]
            )

            # Track best model
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self._save_checkpoint(epoch, is_best=True)

            print(f"Epoch {epoch}/{total_epochs} [Phase 2] - "
                  f"Loss: {train_metrics['total_loss']:.4f}, "
                  f"Train_acc: {train_metrics['accuracy']:.2f}%, "
                  f"Val_acc: {val_metrics['accuracy']:.2f}%")

        return self.best_val_acc

    def _train_epoch_phase1(self, train_loader, epoch, k_disc_steps):
        """Train one epoch of Phase 1 (interleaved D/G)."""
        self.distill_model.train()

        metrics = {
            "disc_loss": 0.0,
            "disc_acc": 0.0,
            "gen_loss": 0.0,
            "fool_rate": 0.0,
            "teacher_pred_mean": 0.0,
            "student_pred_mean": 0.0,
        }
        num_batches = 0

        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(self.device)

            # Train discriminator for k steps
            for _ in range(k_disc_steps):
                self.distill_model.set_discriminator_mode()
                self.optimizer_D.zero_grad()

                outputs = self.distill_model(inputs, mode='discriminator')
                loss_D = outputs['disc_loss']
                loss_D.backward()
                self.optimizer_D.step()

                metrics["disc_loss"] += outputs["disc_loss"].item()
                metrics["disc_acc"] += outputs["disc_accuracy"] * 100
                metrics["teacher_pred_mean"] += outputs["teacher_pred_mean"]

            # Train generator for 1 step
            self.distill_model.set_generator_mode()
            self.optimizer_G.zero_grad()

            outputs = self.distill_model(inputs, mode='generator')
            loss_G = outputs['gen_loss']
            loss_G.backward()
            self.optimizer_G.step()

            metrics["gen_loss"] += outputs["gen_loss"].item()
            metrics["fool_rate"] += outputs["fool_rate"] * 100
            metrics["student_pred_mean"] += outputs["student_pred_mean"]

            num_batches += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
            if key == "disc_loss" or key == "disc_acc" or key == "teacher_pred_mean":
                metrics[key] /= k_disc_steps  # Account for multiple D steps

        return metrics

    def _validate_epoch_phase1(self, val_loader, epoch):
        """Validate one epoch of Phase 1."""
        self.distill_model.eval()

        metrics = {
            "disc_loss": 0.0,
            "disc_acc": 0.0,
            "gen_loss": 0.0,
            "fool_rate": 0.0,
            "teacher_pred_mean": 0.0,
            "student_pred_mean": 0.0,
        }
        num_batches = 0

        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(self.device)

                # Evaluate discriminator
                self.distill_model.set_discriminator_mode()
                disc_outputs = self.distill_model(inputs, mode='discriminator')

                # Evaluate generator
                self.distill_model.set_generator_mode()
                gen_outputs = self.distill_model(inputs, mode='generator')

                metrics["disc_loss"] += disc_outputs["disc_loss"].item()
                metrics["disc_acc"] += disc_outputs["disc_accuracy"] * 100
                metrics["teacher_pred_mean"] += disc_outputs["teacher_pred_mean"]
                metrics["gen_loss"] += gen_outputs["gen_loss"].item()
                metrics["fool_rate"] += gen_outputs["fool_rate"] * 100
                metrics["student_pred_mean"] += gen_outputs["student_pred_mean"]

                num_batches += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        return metrics

    def _train_epoch_phase2(self, train_loader, epoch):
        """Train one epoch of Phase 2 (DKD)."""
        self.distill_model.train()

        metrics = {
            "ce_loss": 0.0,
            "dkd_loss": 0.0,
            "total_loss": 0.0,
            "accuracy": 0.0,
        }
        num_batches = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
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

            metrics["ce_loss"] += ce_loss.item()
            metrics["dkd_loss"] += dkd_loss.item()
            metrics["total_loss"] += total_loss.item()
            num_batches += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        metrics["accuracy"] = 100.0 * correct / total

        return metrics

    def _validate_epoch_phase2(self, val_loader, epoch):
        """Validate one epoch of Phase 2."""
        self.distill_model.eval()

        metrics = {
            "ce_loss": 0.0,
            "dkd_loss": 0.0,
            "total_loss": 0.0,
            "accuracy": 0.0,
        }
        num_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
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

                metrics["ce_loss"] += ce_loss.item()
                metrics["dkd_loss"] += dkd_loss.item()
                metrics["total_loss"] += total_loss.item()
                num_batches += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        metrics["accuracy"] = 100.0 * correct / total

        return metrics

    def _train_standard(self, train_loader, val_loader):
        """Standard training loop for other methods."""
        # TODO: Implement standard training for other methods
        raise NotImplementedError("Standard training not yet implemented. Use DisDKD for now.")

    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        from utils.checkpoint import save_checkpoint

        save_checkpoint(
            self.student.state_dict() if self.student else self.teacher.state_dict(),
            self.args.save_dir,
            self.args.dataset,
            self.args.train_domains if hasattr(self.args, 'train_domains') else None,
            self.args.val_domains if hasattr(self.args, 'val_domains') else None,
            is_best=is_best
        )
