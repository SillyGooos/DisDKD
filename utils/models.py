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
