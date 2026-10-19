"""Schema guards for preserving camera geometry during dataset conversion."""
from __future__ import annotations


def validate_no_crop_camera_schema(schema: dict) -> None:
    """Reject crop or implicit resize requests in a camera conversion schema.

    Conversion must preserve source pixels and resolution. Aspect-preserving
    letterbox belongs to the model input preprocessing stage, not the dataset
    conversion stage.
    """
    cameras = schema.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError("camera schema must define at least one camera")

    for key, spec in cameras.items():
        if "crop" in spec and spec["crop"] not in (None, {}):
            raise ValueError(f"camera {key!r} requests cropping; conversion cropping is forbidden")
        transform = spec.get("transform", "identity")
        if transform != "identity":
            raise ValueError(
                f"camera {key!r} uses transform {transform!r}; only identity is allowed "
                "during conversion (letterbox is applied at model input)"
            )

        source_shape = spec.get("source_shape")
        output_shape = spec.get("output_shape")
        if source_shape is not None and output_shape is not None:
            if list(source_shape) != list(output_shape):
                raise ValueError(
                    f"camera {key!r} changes resolution from {source_shape} to {output_shape}; "
                    "conversion must preserve source resolution"
                )
