from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import os
import glob
import skimage
from skimage.transform import resize
import argparse
import datetime


def resize_bool_mask(mask, target_shape):
    """
    Resize boolean mask safely.
    target_shape: (height, width)
    """
    return resize(
        mask,
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype(bool)


def crop_fixed_size_from_center(image_rgb, center_x, center_y, crop_size, pad_value=255):
    """
    Crop fixed-size square patch from original RGB image.
    If crop goes outside image boundary, pad with white background.

    image_rgb: H x W x 3 RGB image
    center_x, center_y: center of crop
    crop_size: output crop size, e.g. 128
    pad_value: background value for padding
    """

    h, w = image_rgb.shape[:2]
    half = crop_size // 2

    x1 = int(round(center_x)) - half
    y1 = int(round(center_y)) - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # Create white canvas
    crop = np.full(
        (crop_size, crop_size, 3),
        pad_value,
        dtype=image_rgb.dtype
    )

    # Valid region in original image
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    # Corresponding region in crop image
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    crop[dst_y1:dst_y2, dst_x1:dst_x2] = image_rgb[src_y1:src_y2, src_x1:src_x2]

    return crop


def get_mask_center(bool_mask, method="bbox"):
    """
    Get center of a mask.

    method="bbox": use bounding box center
    method="centroid": use region centroid
    """

    ys, xs = np.where(bool_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    if method == "bbox":
        center_x = (xs.min() + xs.max()) / 2
        center_y = (ys.min() + ys.max()) / 2

    elif method == "centroid":
        center_x = xs.mean()
        center_y = ys.mean()

    else:
        raise ValueError("method should be 'bbox' or 'centroid'")

    return center_x, center_y


def segment_images(args, image_name, destination_folder):
    image_path = os.path.join(root_folder, image_name)
    print(f"Name: {image_name}")

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        print(f"Failed to read image, skipped: {image_path}")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_orig = image_rgb.copy()

    image_for_sam = image_rgb.copy()

    if args.isresize:
        width = int(image_for_sam.shape[1] * args.resize_factor)
        height = int(image_for_sam.shape[0] * args.resize_factor)
        image_for_sam = cv2.resize(
            image_for_sam,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    os.makedirs(destination_folder, exist_ok=True)

    base_name = os.path.splitext(image_name)[0]

    all_crops_name = os.path.join(
        destination_folder,
        f"{base_name}_crop_*.png"
    )

    processed_imgs = glob.glob(all_crops_name)

    if len(processed_imgs) > 0 and not args.overwrite:
        print("Image already processed, skipped.")
        return

    masks = mask_generator.generate(image_for_sam)
    print(f"Masks detected by SAM: {len(masks)}")

    if len(masks) == 0:
        print("No masks detected.")
        return

    area_thresh = image_for_sam.shape[0] * image_for_sam.shape[1] * args.area_thresh_ratio
    print(f"Area threshold: {area_thresh}")

    saved_count = 0

    for i, mask_info in enumerate(masks):
        bool_mask = mask_info["segmentation"]

        if args.isresize:
            bool_mask = resize_bool_mask(
                bool_mask,
                (image_orig.shape[0], image_orig.shape[1])
            )
            area = np.sum(bool_mask)
            area_thresh_original = image_orig.shape[0] * image_orig.shape[1] * args.area_thresh_ratio
        else:
            bool_mask = bool_mask.astype(bool)
            area = mask_info["area"]
            area_thresh_original = area_thresh

        if area < area_thresh_original:
            continue

        center = get_mask_center(bool_mask, method=args.center_method)

        if center is None:
            continue

        center_x, center_y = center

        crop_rgb = crop_fixed_size_from_center(
            image_rgb=image_orig,
            center_x=center_x,
            center_y=center_y,
            crop_size=args.crop_size,
            pad_value=0
        )

        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

        save_path = os.path.join(
            destination_folder,
            f"{base_name}_crop_{i}.png"
        )

        ok = cv2.imwrite(save_path, crop_bgr)

        if ok:
            saved_count += 1
            print(
                f"Saved crop {saved_count}: {save_path}, "
                f"area={area}, center=({center_x:.1f}, {center_y:.1f})"
            )
        else:
            print(f"Failed to save: {save_path}")

    print(f"Total crops saved for {image_name}: {saved_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AutoSeg Fixed Size Crop")

    parser.add_argument(
        "--isresize",
        action="store_true",
        help="Resize image before SAM segmentation to increase speed."
    )

    parser.add_argument(
        "--resize_factor",
        type=float,
        default=0.5,
        help="Resize factor for SAM input."
    )

    parser.add_argument(
        "--area_thresh_ratio",
        type=float,
        default=0.00001,
        help="Minimum mask area ratio."
    )

    parser.add_argument(
        "--crop_size",
        type=int,
        default=128,
        help="Fixed crop size. All output crops will be crop_size x crop_size."
    )

    parser.add_argument(
        "--center_method",
        type=str,
        default="bbox",
        choices=["bbox", "centroid"],
        help="How to define crop center. bbox is usually more stable."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results."
    )

    args = parser.parse_args()

    root_folder = "/Users/u5644731/Documents/bioimage_hakathon/origin"
    destination_folder = "/Users/u5644731/Documents/bioimage_hakathon/cropped"

    ckpt_vit_b = "/Users/u5644731/Downloads/sam_vit_b_01ec64.pth"

    sam = sam_model_registry["vit_b"](checkpoint=ckpt_vit_b)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        pred_iou_thresh=0.6,
        stability_score_thresh=0.6,
        crop_n_layers=1,
        crop_n_points_downscale_factor=4,
        min_mask_region_area=5
    )

    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    for image_name in os.listdir(root_folder):
        if not image_name.lower().endswith(valid_exts):
            print(f"Skip non-image file: {image_name}")
            continue

        image_path = os.path.join(root_folder, image_name)

        if not os.path.isfile(image_path):
            print(f"Skip folder: {image_path}")
            continue

        print(datetime.datetime.now())
        segment_images(args, image_name, destination_folder)