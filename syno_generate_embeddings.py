import logging

import torch

import utils

# create logger
logger = logging.getLogger(__name__)


def remove_deleted_images(emb_file: str, image_id_path_dict: dict):
    logger.info(f"Identifying deleted images in {emb_file}.")
    # read file
    images = utils.read_emb_file(emb_file)
    if not images:
        logger.info(f"No images in {emb_file}.")
        return
    ids, paths, emb = images["ids"], images["paths"], images["embeddings"]
    ind_to_keep = [
        ind for ind, image_id in enumerate(ids) if image_id in image_id_path_dict
    ]

    deleted = len(ids) - len(ind_to_keep)
    if deleted:
        ids = [ids[ind] for ind in ind_to_keep]
        paths = [paths[ind] for ind in ind_to_keep]
        emb = emb[ind_to_keep, :]
        logger.info(f"Removed {deleted} deleted images from {emb_file}.")
        # write updated file
        utils.overwrite_emb_file(emb_file, ids, paths, emb)
    return


def get_new_images(
    image_id_path_dict: dict, scanned_ids: list, n_files: int = None
) -> dict:
    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        # '.gif',
        ".heic",
    )
    new_images = dict()

    if n_files is not None and n_files <= 0:
        logger.info(f"Requested {n_files} new images. None will be retrieved.")
        return new_images
    logger.debug(f"Collecting {n_files} new images...")

    scanned_ids = set(scanned_ids)
    for image_id in image_id_path_dict:
        if n_files is not None and len(new_images) >= n_files:
            break
        if (
            image_id_path_dict[image_id].lower().endswith(image_extensions)
            and image_id not in scanned_ids
        ):
            new_images[image_id] = image_id_path_dict[image_id]
    logger.debug(f"Collected {len(new_images)} new images.")
    return new_images


def add_embeddings(
    model,
    emb_file: str,
    image_id_path_dict: dict,
    emb_batch: int,
    nn_batch: int = 1,
    max_images: int = None,
):
    scanned_images = utils.read_emb_file(emb_file)
    if not scanned_images:
        scanned_ids, scanned_paths, scanned_emb = [], [], None
    else:
        scanned_ids, scanned_paths, scanned_emb = (
            scanned_images["ids"],
            scanned_images["paths"],
            scanned_images["embeddings"],
        )
    if emb_batch and (max_images is None or max_images > len(scanned_ids)):
        if max_images is not None:
            n_files = max_images - len(scanned_images)
            logger.info(f"Scanning {n_files} new images...")
        else:
            n_files = None
            logger.info(f"Scanning for all new images...")
    else:
        logger.info(
            f"Requested total {max_images} images and {emb_file} already has {len(scanned_ids)} images. No new images will be embedded."
        )
        return scanned_ids, scanned_paths, scanned_emb
    new_images_dict = get_new_images(image_id_path_dict, scanned_ids, n_files)
    n_new_images = len(new_images_dict)
    new_images_embedded = 0
    while len(new_images_dict):
        images_in_batch = min(emb_batch, len(new_images_dict))
        logger.info(
            f"PIL reading {new_images_embedded + 1}th to {new_images_embedded + images_in_batch}th images of {n_new_images} new images..."
        )
        pil_images = []

        while len(pil_images) < images_in_batch:
            image_id, image_path = new_images_dict.popitem()
            pil_img = utils.pil_read_images(
                utils.get_absolute_path_in_container(image_path, validate_path=True)
            )
            if pil_img is not None:
                pil_images.append(pil_img)
                scanned_ids.append(image_id)
                scanned_paths.append(image_path)
            else:
                new_images_dict[image_id] = image_path
        new_images_embedded += images_in_batch

        logger.info(f"Encoding {images_in_batch} images...")
        new_img_emb = model.encode(
            pil_images,
            batch_size=nn_batch,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

        if scanned_emb is not None:
            scanned_emb = torch.cat((scanned_emb, new_img_emb), 0)
        else:
            scanned_emb = new_img_emb
        logger.info(
            f"Encoded {images_in_batch} new images for total of {len(scanned_images)} images."
        )

        utils.overwrite_emb_file(emb_file, scanned_ids, scanned_paths, scanned_emb)

    return (scanned_ids, scanned_paths, scanned_emb)


def update_embeddings(
    model,
    emb_file: str,
    image_id_path_dict: dict,
    torch_threads: int,
    emb_batch: int,
    nn_batch: int = 1,
    max_images: int = None,
):
    torch.set_num_threads(torch_threads)

    remove_deleted_images(emb_file=emb_file, image_id_path_dict=image_id_path_dict)
    _ = add_embeddings(
        model=model,
        emb_file=emb_file,
        image_id_path_dict=image_id_path_dict,
        emb_batch=emb_batch,
        nn_batch=nn_batch,
        max_images=max_images,
    )
