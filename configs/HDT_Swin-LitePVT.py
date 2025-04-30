import torch
import torch.nn as nn
try:
    # Import MMDetection registry and BaseBackbone if available
    from mmdet.models.builder import BACKBONES
    from mmdet.models.backbones.base_backbone import BaseBackbone
except ImportError:
    # If MMDetection is not installed, define BaseBackbone as an alias for nn.Module for standalone usage
    BaseBackbone = nn.Module

try:
    import timm  # TIMM library provides pre-trained vision transformers
except ImportError:
    timm = None  # If timm is not installed, the user must install it or provide backbone implementations

# Register the backbone in MMDetection's registry if available
@ (BACKBONES.register_module() if 'BACKBONES' in globals() else lambda cls: cls)
class DualTransformerBackbone(BaseBackbone):
    """Dual-branch Transformer backbone with Swin Transformer (primary) and a lightweight Transformer (secondary).
    
    This backbone runs a Swin Transformer and a lightweight Transformer (e.g., PVT or MobileViT) in parallel.
    Their multi-scale feature maps are fused at each corresponding stage. The output is a tuple of fused features 
    at different stages, compatible with detection necks (e.g., FPN). Hooks for knowledge distillation and pruning 
    are marked for future use.
    """
    def __init__(self, 
                 swin_type: str = 'swin_tiny_patch4_window7_224',
                 light_type: str = 'pvt_tiny', 
                 pretrained_swin: str = None,
                 pretrained_light: str = None,
                 out_indices: tuple = (0, 1, 2, 3),
                 fuse_method: str = 'sum',
                 init_cfg: dict = None):
        """
        Args:
            swin_type (str): Model name for the Swin Transformer backbone (primary branch) in TIMM.
            light_type (str): Model name for the lightweight Transformer backbone (secondary branch) in TIMM.
            pretrained_swin (str, optional): Path to pretrained weights for the Swin backbone (if any).
            pretrained_light (str, optional): Path to pretrained weights for the lightweight backbone (if any).
            out_indices (tuple): Indices of stages to output features from (default corresponds to all stages).
            fuse_method (str): Fusion method for combining features ('sum' or 'concat'). Default is 'sum'.
            init_cfg (dict, optional): Initialization config for BaseBackbone (unused if using TIMM pretrained weights).
        """
        super(DualTransformerBackbone, self).__init__(init_cfg=init_cfg)
        assert timm is not None, "TIMM library is required for DualTransformerBackbone"
        
        # Initialize primary backbone (Swin Transformer) with TIMM
        self.swin_backbone = timm.create_model(
            swin_type, pretrained=(pretrained_swin is None and True or False),
            features_only=True, out_indices=out_indices
        )
        if pretrained_swin:
            # Load custom pretrained weights if provided
            state_dict = torch.load(pretrained_swin, map_location='cpu')
            self.swin_backbone.load_state_dict(state_dict, strict=False)
        # Primary backbone (Swin) is now ready
        
        # Initialize secondary backbone (lightweight Transformer) with TIMM
        self.light_backbone = timm.create_model(
            light_type, pretrained=(pretrained_light is None and True or False),
            features_only=True, out_indices=out_indices
        )
        if pretrained_light:
            # Load custom pretrained weights if provided
            state_dict = torch.load(pretrained_light, map_location='cpu')
            self.light_backbone.load_state_dict(state_dict, strict=False)
        # Secondary backbone (lightweight) is now ready
        
        # Ensure both backbones have the same number of output stages
        self.out_indices = out_indices
        num_stages = len(out_indices)
        assert len(self.swin_backbone.feature_info) == num_stages, \
            "Swin backbone output stages mismatch out_indices length"
        assert len(self.light_backbone.feature_info) == num_stages, \
            "Light backbone output stages mismatch out_indices length"
        
        # Get channel dimensions of features at each stage for both backbones
        swin_channels = list(self.swin_backbone.feature_info.channels())
        light_channels = list(self.light_backbone.feature_info.channels())
        assert len(swin_channels) == len(light_channels) == num_stages
        
        # Define convolution layers to align channel dimensions for fusion (if needed)
        self.fusion_convs = nn.ModuleList()
        for i in range(num_stages):
            if swin_channels[i] != light_channels[i]:
                # If channels differ, add a 1x1 conv to map the light branch's channels to match the Swin's channels
                self.fusion_convs.append(nn.Conv2d(light_channels[i], swin_channels[i], kernel_size=1))
            else:
                # If already matching, use identity mapping
                self.fusion_convs.append(nn.Identity())
        
        # Store fusion method (either 'sum' or 'concat')
        assert fuse_method in ('sum', 'concat'), "fuse_method must be 'sum' or 'concat'"
        self.fuse_method = fuse_method
        
        # Hook placeholders (no actual functionality, just markers for future extension)
        # hook: You can integrate pruning mechanisms here (e.g., replace parts of swin_backbone with pruned versions)
        # hook: You can integrate knowledge distillation loss computation in the training loop using this backbone
    
    def forward(self, x):
        """
        Forward pass through both primary (Swin) and secondary (lightweight) backbones, with feature fusion.
        
        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).
        
        Returns:
            tuple[Tensor]: Tuple of fused feature maps from multiple stages. Each Tensor is the fused output 
                           of corresponding stages from the Swin and lightweight backbones.
        """
        # Forward through Swin Transformer (primary branch)
        swin_feats = self.swin_backbone(x)   # list of feature maps from Swin at indices specified in out_indices
        # Forward through lightweight Transformer (secondary branch)
        light_feats = self.light_backbone(x)  # list of feature maps from light backbone at the same stages
        
        # Ensure we have the same number of features from each branch
        assert len(swin_feats) == len(light_feats) == len(self.out_indices)
        
        fused_feats = []
        # Fuse features from both branches at each stage
        for i, (feat_swin, feat_light) in enumerate(zip(swin_feats, light_feats)):
            # Align channel dimensions if needed using the pre-defined conv (or identity)
            if feat_light.shape[1] != feat_swin.shape[1]:
                feat_light = self.fusion_convs[i](feat_light)
            # Fuse the feature maps
            if self.fuse_method == 'sum':
                # Element-wise sum fusion
                fused = feat_swin + feat_light
            elif self.fuse_method == 'concat':
                # Concatenate along the channel dimension
                fused = torch.cat([feat_swin, feat_light], dim=1)
                # (Optionally, a 1x1 conv could follow concatenation to reduce channels, but not included by default)
            else:
                # Default fallback (should not happen if fuse_method is validated)
                fused = feat_swin + feat_light
            fused_feats.append(fused)
        
        # At this point, fused_feats is a list of feature maps from each stage
        # hook: KD loss can be computed here using swin_feats and light_feats for training (if implementing knowledge distillation)
        # hook: Additional pruning steps can be applied here to fused_feats or the branch features if needed
        
        # Return fused feature maps as a tuple (for compatibility with MMDetection necks)
        return tuple(fused_feats)
