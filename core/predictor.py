# core/predictor.py

import torch

class Predictor:
    def __init__(self, model, device):
        """
        model: nn.Module entraîné
        device: 'cuda' ou 'cpu'
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def predict(self, inputs, return_probs=False):
        """
        inputs: image (1xC*H*W) ou batch (B x C x H x W), tensor déjà préprocessé
        return_probs: si True, retourne les probabilités (softmax)
        Returns:
            - indices des classes (par défaut)
            - ou probabilités si return_probs=True
        """
        with torch.no_grad():
            inputs = inputs.to(self.device, non_blocking=True)
            outputs = self.model(inputs)

            if return_probs:
                probs = torch.softmax(outputs, dim=1)
                return probs

            _, predicted = torch.max(outputs, 1)
            return predicted
