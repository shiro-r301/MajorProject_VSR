import diffusers
from temporal_cogvideox_transformer_3d_batch_proccessing import CogVideoXTransformer3DModel, TemporalCurvatureGuidance

def patch_CogVideoXTransformer3DModel():
    diffusers.CogVideoXTransformer3DModel = CogVideoXTransformer3DModel
    diffusers.models.transformers.cogvideox_transformer_3d.CogVideoXTransformer3DModel = CogVideoXTransformer3DModel
    print("Successfully Patched code to temporal_cogVideoX!\n")


