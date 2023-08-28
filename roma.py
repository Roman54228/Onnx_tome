import torch
import onnxruntime
import timm
import tome

# Load a pretrained model, can be any vit / deit model.
# model = timm.create_model("vit_base_patch16_224", pretrained=False)
# # Patch the model with ToMe.
# tome.patch.timm(model)
# # Set the number of tokens reduced per layer. See paper for details.
# model.r = 1

import torch, tome


# Load a pretrained model, can be any vit / deit model.
model = timm.create_model("vit_base_patch16_224", pretrained=False)
# Patch the model with ToMe.
tome.patch.timm(model)
# Set the number of tokens reduced per layer. See paper for details.
model.r = 16  

x = torch.randn(1, 3, 224, 224, requires_grad=True)
torch_out = model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "kek.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=15,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
