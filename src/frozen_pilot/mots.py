"""Read-only MOTS20 text/RLE adapter. No dataset conversion or writes."""
from dataclasses import dataclass
from pathlib import Path
import configparser
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


def encode(mask):
    return coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))


@dataclass
class PersonGT:
    object_id: int
    rle: dict

    def decode(self):
        return coco_mask.decode(self.rle).astype(bool)


@dataclass
class Frame:
    sequence: str
    number: int
    image: Path
    height: int
    width: int
    persons: list
    ignore_rle: dict
    ignore_count: int


def load_frames(train_root, sequence, numbers):
    train_root = Path(train_root).resolve()
    if sequence not in {"MOTS20-02", "MOTS20-05", "MOTS20-09", "MOTS20-11"}:
        raise ValueError("Expected a MOTS20 train sequence")
    if len(set(numbers)) != len(numbers) or not numbers:
        raise ValueError("Frame numbers must be unique and nonempty")
    base = train_root / sequence
    config = configparser.ConfigParser()
    with (base / "seqinfo.ini").open() as stream:
        config.read_file(stream)
    meta = config["Sequence"]
    if meta["name"] != sequence:
        raise ValueError("Sequence metadata/name mismatch")
    h, w = int(meta["imHeight"]), int(meta["imWidth"])
    n = int(meta["seqLength"])
    if any(not 1 <= i <= n for i in numbers):
        raise ValueError("Frame out of sequence bounds")
    groups = {i: {"persons": [], "ignore": [], "ids": set()} for i in numbers}
    with (base / "gt/gt.txt").open() as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 6:
                raise ValueError("MOTS annotation must have six fields")
            frame, oid, category, mh, mw = map(int, fields[:5])
            if frame not in groups:
                continue
            if (mh, mw) != (h, w) or category not in {2, 10}:
                raise ValueError("Unexpected GT dimensions/class")
            group = groups[frame]
            if oid in group["ids"]:
                raise ValueError("Duplicate frame/object ID")
            group["ids"].add(oid)
            rle = {"size": [h, w], "counts": fields[5].encode("ascii")}
            mask = coco_mask.decode(rle)
            if mask.shape != (h, w) or not mask.any():
                raise ValueError("Invalid or empty GT mask")
            if category == 2:
                group["persons"].append(PersonGT(oid, rle))
            else:
                group["ignore"].append(rle)
    frames = []
    for number in sorted(numbers):
        image = base / meta["imDir"] / f"{number:06d}{meta['imExt']}"
        with Image.open(image) as im:
            im.load()
            if im.size != (w, h):
                raise ValueError(f"Source/GT dimensions disagree: {image}")
        group = groups[number]
        ignore = (coco_mask.merge(group["ignore"]) if group["ignore"]
                  else encode(np.zeros((h, w), dtype=bool)))
        frames.append(Frame(sequence, number, image, h, w,
                            sorted(group["persons"], key=lambda g: g.object_id),
                            ignore, len(group["ignore"])))
    return frames


def validate_known_gt(frame):
    if frame.sequence != "MOTS20-02" or frame.number != 1:
        raise ValueError("Known GT validation requires MOTS20-02 frame 1")
    person = next(g for g in frame.persons if g.object_id == 2002)
    mask = person.decode()
    actual = {"image_size_wh": [frame.width, frame.height],
              "mask_shape_hw": list(mask.shape), "area": int(mask.sum()),
              "bbox_xywh": coco_mask.toBbox(person.rle).tolist()}
    expected = {"image_size_wh": [1920, 1080], "mask_shape_hw": [1080, 1920],
                "area": 28863, "bbox_xywh": [1340, 421, 169, 368]}
    if actual != expected:
        raise RuntimeError(f"STOP: known GT mismatch: {actual} != {expected}")
    return actual
