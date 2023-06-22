import logging
import os
import pickle
import sys

import PIL
import torch
from PIL import Image
from pillow_heif import register_heif_opener

# for reading heic images
register_heif_opener()

# create logger
logger = logging.getLogger(__name__)


def validated_path(filepath: str):
    if os.path.exists(filepath):
        return filepath
    # change extension case
    filename, file_extension = os.path.splitext(filepath)
    if file_extension.islower():
        new_filepath = filename + file_extension.upper()
    else:
        new_filepath = filename + file_extension.lower()
    if os.path.exists(new_filepath):
        logger.debug(f"{filepath} was not found, but found {new_filepath}.")
        return new_filepath
    logger.info(
        f"File path {filepath} is in Synology image tree, but could not find on disk."
    )
    return None


def get_absolute_path_in_container(rel_path: str, validate_path: bool = False):
    photos_dir = os.environ.get("DEST_MOUNT_DIR")
    if photos_dir is not None:
        if validate_path:
            # extension case difference can exist between synology db path and actual path
            return validated_path(os.path.join(photos_dir, rel_path))
        return os.path.join(photos_dir, rel_path)
    message = "DEST_MOUNT_DIR is not in environment."
    logger.error(message)
    raise ValueError(message)


def underlined_choices(choices: list[str]) -> tuple[str, dict[str, str]]:
    choices_string_list = []
    underlines = dict()
    for s in choices:
        underline_success = False
        for i in range(len(s)):
            if s[: (i + 1)] not in underlines:
                choices_string_list.append(
                    "\033[4m" + s[: (i + 1)] + "\033[0m" + s[(i + 1) :]
                )
                underlines[s[: (i + 1)]] = s
                underline_success = True
                break

        if not underline_success:
            logger.debug(
                "Due to similarity of some choices, an underlined short-form could not be constructed."
            )
            choices_string_list = choices
            underlines = {s: s for s in choices}
            break
    choices_string = " | ".join(choices_string_list)
    return (choices_string, underlines)


def get_user_choice(choices: list[str], prompt: str):
    choices = [str(s).strip().lower() for s in choices]
    choices_string, underlines = underlined_choices(choices)
    user_choice = input(f"{prompt} ({choices_string}) : ")
    while True:
        user_choice = user_choice.strip().lower()
        if user_choice in choices:
            return user_choice
        if user_choice in underlines:
            return underlines[user_choice]
        user_choice = input(f"Please select a valid option ({choices_string}) : ")


def read_emb_file(emb_file: str):
    if os.path.exists(emb_file):
        with open(emb_file, "rb") as fIn:
            scanned_images = pickle.load(fIn)
            if not scanned_images:
                msg = f"{emb_file} exists but does not contain any images."
                logger.error(msg)
                raise ValueError(msg)
            scanned_ids, scanned_paths, scanned_emb = (
                scanned_images["ids"],
                scanned_images["paths"],
                scanned_images["embeddings"],
            )
            if len(scanned_ids) != len(scanned_paths) or len(
                scanned_ids
            ) != scanned_emb.size(dim=0):
                msg = f"Length of images does not match length of embeddings in {emb_file}. File is likely corrupted, must be deleted, and recreated."
                logger.error(msg)
                raise ValueError(msg)
            logger.info(
                f"{emb_file} exists and contains embeddings for {len(scanned_ids)} images."
            )
    else:
        logger.info(f"{emb_file} does not exist.")
        return None

    return scanned_images


def overwrite_emb_file(
    emb_file: str, scanned_ids: list, scanned_paths: list, scanned_emb: torch.Tensor
):
    if len(scanned_ids) != len(scanned_paths) or len(scanned_ids) != scanned_emb.size(
        dim=0
    ):
        msg = f"Length of images does not match length of embeddings to write out. {emb_file} will not be written out to prevent corruption."
        logger.error(msg)
        raise ValueError(msg)
    if (not len(scanned_ids)) or (not len(scanned_paths)):
        logger.info(
            f"Provided 0 images. {emb_file} will not be written out to prevent corruption."
        )
    with open(emb_file, "wb") as fOut:
        scanned_images = dict(
            ids=scanned_ids, paths=scanned_paths, embeddings=scanned_emb
        )
        pickle.dump(scanned_images, fOut, pickle.HIGHEST_PROTOCOL)
        logger.info(f"{emb_file} written out with {len(scanned_ids)} images.")
    return


def pil_read_images(image_path: str):
    try:
        img = Image.open(image_path)
    except PIL.UnidentifiedImageError:
        logger.warn(
            f"Image could not be read and will be retried in next batch - {image_path}"
        )
        img = None
    return img


def get_log_level(verbose: int = 0) -> int:
    # Define log levels
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    # Set log level based on verbosity input
    if verbose >= len(levels):
        level = logging.DEBUG
    else:
        level = levels[verbose]

    return level


def configure_logger(
    logger: logging.Logger,
    log_file: str = None,
    verbose: int = 1,
    formatter: logging.Formatter = None,
):
    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(funcName)s - %(message)s"
        )
    stdout_log_level = logging.INFO

    logger.setLevel(stdout_log_level)
    # Create stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(stdout_log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        logger.info(f"Logs with verbosity={verbose} will be recorded in {log_file}.")
        # Create file handler
        file_log_level = get_log_level(verbose)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # update logger's log level
        if file_log_level < stdout_log_level:
            logger.setLevel(file_log_level)
