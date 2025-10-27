import torchvision.models.optical_flow as of_models

# Load pre-trained RAFT model
raft_model = of_models.raft_large(pretrained=True).eval()
