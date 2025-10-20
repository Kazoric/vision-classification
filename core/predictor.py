# Import necessary libraries
import torch
from torch import nn

class Predictor:
    """
    Class for making predictions using a trained model.
    
    Attributes:
        model (nn.Module): Trained model to use for prediction
        device (str): Device to use for prediction (e.g., 'cuda' or 'cpu')
    """
    
    def __init__(self, model: nn.Module, device: str) -> None:
        """
        Initialize the Predictor object.
        
        Args:
            model (nn.Module): Trained model to use for prediction
            device (str): Device to use for prediction
        
        Returns:
            None
        """
        
        # Set model and device attributes
        self.model = model
        self.device = device
        
        # Move model to specified device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()

    def predict(self, inputs: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """
        Make predictions on a given input.
        
        Args:
            inputs (torch.Tensor): Input tensor (e.g., image or batch of images)
            return_probs (bool): Whether to return class probabilities instead of indices
        
        Returns:
            torch.Tensor: Predicted class indices or probabilities
        """
        
        # Disable gradient computation for prediction
        with torch.no_grad():
            
            # Move input to specified device
            inputs = inputs.to(self.device, non_blocking=True)
            
            # Make predictions using model
            outputs = self.model(inputs)

            # Return class probabilities if requested
            if return_probs:
                probs = torch.softmax(outputs, dim=1)
                return probs

            # Otherwise, return predicted class indices
            _, predicted = torch.max(outputs, 1)
            return predicted
