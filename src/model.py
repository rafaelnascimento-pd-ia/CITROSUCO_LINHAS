import segmentation_models_pytorch as smp

def get_segmentation_model(encoder_name="resnet18", encoder_weights="imagenet", 
                           in_channels=3, classes=1):
    """
    Creates and returns a UNet++ segmentation model using the specified encoder.
    Supported encoders for lightweight GPU (RTX 2050):
      - resnet18 (Default, very stable)
      - mobilenet_v2 (Very lightweight)
      - efficientnet-b0 (Good accuracy/speed trade-off)
    """
    supported_encoders = ["resnet18", "mobilenet_v2", "efficientnet-b0"]
    if encoder_name not in supported_encoders:
        raise ValueError(f"Encoder '{encoder_name}' não suportado! Escolha entre: {supported_encoders}")
        
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None  # We use raw logits. Sigmoid is applied during loss/inference.
    )
    return model

if __name__ == "__main__":
    model = get_segmentation_model("resnet18")
    print(f"UNet++ model with ResNet18 encoder instantiated successfully.")
