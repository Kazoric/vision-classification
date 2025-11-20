import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Cleaned version of Grad-CAM for CNNs and ViTs.
    """
    def __init__(self, model, target_layer):
        # Assuming model is your wrapper, we access the architecture via .model
        self.model = model.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook registration
        # Note: 'module', 'input', 'grad_input' are enforced by PyTorch even if unused.
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _reshape_transform(self, tensor):
        """
        Handles reshaping for ViT: (B, N, D) -> (B, D, H, W)
        Removed unnecessary height/width args.
        """
        # Standard CNN case: (B, C, H, W) -> No change needed
        if len(tensor.shape) == 4:
            return tensor

        # ViT case: (B, N, D)
        if len(tensor.shape) == 3:
            # Remove CLS token (index 0)
            spatial_tokens = tensor[:, 1:, :] 
            
            # Calculate H and W from the number of patches
            # N_patches = H * W. Assuming a square image.
            num_patches = spatial_tokens.shape[1]
            h = w = int(np.sqrt(num_patches))
            
            if h * w != num_patches:
                raise ValueError(f"Dimension error: {num_patches} patches is not a perfect square.")

            # Transpose: (B, N, D) -> (B, D, N) -> (B, D, H, W)
            result = spatial_tokens.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], h, w)
            return result
            
        raise ValueError(f"Unknown tensor shape: {tensor.shape}")

    def generate(self, input_tensor, target_class=None):
        # 1. Forward Pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # 2. Backward Pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 3. Retrieve data via hooks
        grads = self.gradients
        acts = self.activations

        # 4. Reshape if necessary (ViT)
        # No need to pass image size here anymore
        acts = self._reshape_transform(acts)
        grads = self._reshape_transform(grads)

        # 5. Grad-CAM calculation
        # Global Average Pooling on gradients (Alpha weights)
        weights = torch.mean(grads, dim=(2, 3), keepdim=True) # (1, C, 1, 1)

        # Linear combination of activations by weights
        cam = torch.sum(weights * acts, dim=1).squeeze() # (H, W)

        # ReLU (keep only positive influence)
        cam = F.relu(cam)

        # Min-Max normalization to get an image between 0 and 1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.detach().cpu().numpy()

def overlay_heatmap(img_path, cam_mask, alpha=0.5, resize_dim=(224, 224)):
    """Helper to draw the heatmap."""
    img = cv2.imread(img_path)
    
    # Added resize_dim parameter to avoid hardcoding 224x224
    if resize_dim:
        img = cv2.resize(img, resize_dim)
        
    # Resize the mask to the size of the loaded image
    heatmap = cv2.resize(cam_mask, (img.shape[1], img.shape[0]))
    
    # Colorization
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blending
    result = heatmap * alpha + img * (1 - alpha)
    return np.uint8(result)