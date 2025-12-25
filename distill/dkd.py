import torch
import torch.nn as nn
import torch.nn.functional as F


class DKD(nn.Module):
    """
    Decoupled Knowledge Distillation (CVPR 2022).
    
    Decouples the KD loss into Target Class Knowledge Distillation (TCKD)
    and Non-Target Class Knowledge Distillation (NCKD) components.
    
    Args:
        teacher (nn.Module): Pretrained teacher network.
        student (nn.Module): Student network.
        alpha (float): Weight for TCKD loss.
        beta (float): Weight for NCKD loss.
        temperature (float): Temperature for softmax.
    """
    def __init__(self, teacher, student, alpha=1.0, beta=8.0, temperature=4.0):
        super(DKD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def forward(self, x, targets=None):
        """
        Forward pass that computes both teacher and student outputs.
        
        Args:
            x (Tensor): Input tensor.
            targets (Tensor, optional): Ground truth labels (required for DKD loss).
            
        Returns:
            tuple: (teacher_logits, student_logits, dkd_loss)
                   dkd_loss is None if targets are not provided.
        """
        # Forward pass through teacher and student networks
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        student_logits = self.student(x)
        
        # Compute DKD loss if targets are provided
        if targets is not None:
            dkd_loss = self.compute_dkd_loss(student_logits, teacher_logits, targets)
        else:
            dkd_loss = None
        
        return teacher_logits, student_logits, dkd_loss
    
    def compute_dkd_loss(self, logits_student, logits_teacher, target):
        """
        Compute the Decoupled Knowledge Distillation loss.
        
        Args:
            logits_student (Tensor): Student logits.
            logits_teacher (Tensor): Teacher logits.
            target (Tensor): Ground truth labels.
            
        Returns:
            Tensor: Combined DKD loss (TCKD + NCKD).
        """
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        
        # Compute softmax probabilities
        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        
        # Target Class Knowledge Distillation (TCKD)
        pred_student_tckd = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher_tckd = self._cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student_tckd = torch.log(pred_student_tckd)
        
        tckd_loss = (
            F.kl_div(log_pred_student_tckd, pred_teacher_tckd, reduction='batchmean')
            * (self.temperature ** 2)
        )
        
        # Non-Target Class Knowledge Distillation (NCKD)
        # Mask out the ground truth class by subtracting a large value
        pred_teacher_nckd = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_nckd = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, dim=1
        )
        
        nckd_loss = (
            F.kl_div(log_pred_student_nckd, pred_teacher_nckd, reduction='batchmean')
            * (self.temperature ** 2)
        )
        
        # Combined DKD loss
        return self.alpha * tckd_loss + self.beta * nckd_loss
    
    def _get_gt_mask(self, logits, target):
        """Create mask for ground truth class."""
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask
    
    def _get_other_mask(self, logits, target):
        """Create mask for non-ground truth classes."""
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask
    
    def _cat_mask(self, t, mask1, mask2):
        """Concatenate masked probabilities."""
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt
