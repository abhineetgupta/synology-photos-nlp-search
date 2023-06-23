import argparse
import logging
import os

import torch
from sentence_transformers import SentenceTransformer, util

import utils
from syno_generate_embeddings import update_embeddings
from synoapi import SynoPhotosSharedAPI

# create logger
logger = logging.getLogger(__name__)


def load_model():
    model_name = os.environ.get("ST_MODEL")
    if model_name is not None:
        return SentenceTransformer(model_name)
    message = "ST_MODEL is not in environment."
    logger.error(message)
    raise ValueError(message)


def create_tag_image_dict(
    syno: SynoPhotosSharedAPI, tag_names: list[str] = None
) -> dict[str, dict]:
    # append image ids for a dict of tag names
    logger.info(f"Collecting image list for tags in Synology Photos...")
    images = syno.get_image_list(additional=syno.list_param(["tag"]))
    if not images:
        logger.info("No images were found.")
        return None

    if not tag_names:
        # use all tags
        all_tags = syno.get_tag_list()
        if not all_tags:
            logger.info("No tags were found.")
            return None
        tag_names = [t["name"] for t in all_tags]

    tag_names = set(tag_names)
    result = {}
    # assign image ids to tag ids
    for image in images:
        for tag in image["additional"]["tag"]:
            if tag["name"] in tag_names:
                if tag["name"] not in result:
                    result[tag["name"]] = dict(image_ids=[], id=tag["id"])
                result[tag["name"]]["image_ids"].append(image["id"])

    logger.info(f"Collected image list for tags in Synology Photos.")
    return result


def create_image_path_tree(syno: SynoPhotosSharedAPI) -> dict[int, str]:
    # get files with their filepaths as dict{filepath:id}
    logger.info(f"Collecting image list in Synology Photos...")
    images = syno.get_image_list(additional=syno.list_param(["folder"]))
    if not images:
        logger.info("No images were found.")
        return None
    result = {
        image["id"]: os.path.join(
            image["additional"]["folder"].strip("/"), image["filename"]
        )
        for image in images
    }
    logger.info(f"Collected image list in Synology Photos.")
    return result


def get_user_approval(request: str):
    if request == "update_embeddings":
        result = utils.get_user_choice(
            choices=["yes", "no"],
            prompt="Would you like to update image index to include recent images in search?",
        )
    elif request == "search_embeddings":
        result = utils.get_user_choice(
            choices=["yes", "no"], prompt="Would you like to search through images?"
        )
    else:
        message = f"`{request}` is not a valid request option."
        logger.error(message)
        raise ValueError(message)
    if result == "yes":
        return True
    return False


def parse_query(query: str):
    if not query:
        return None, None
    # split by comma
    tokens = query.rsplit(",", 1)
    if (
        len(tokens) <= 1
        or (not tokens[1].strip().isnumeric())
        or (int(tokens[1].strip()) <= 0)
    ):
        return query.strip(), None
    user_k = int(tokens[1].strip())
    if user_k <= 0:
        logger.warn(
            f"User specified k={user_k}. Instead its default value will be used."
        )
    return tokens[0].strip(), user_k


def get_user_query():
    default_k = os.environ.get("DEFAULT_K")
    if default_k is None:
        message = "DEFAULT_K is not in environment."
        logger.error(message)
        raise ValueError(message)

    query = input(
        f"Enter query and optionally the number of desired images (default={default_k}) separated by comma. Leave empty to exit : "
    )
    query, k = parse_query(query)
    k = k or int(default_k)
    return query, k


def search(query: str, model, image_ids: list, embeddings: torch.Tensor, k: int):
    # First, we encode the query (which can either be an image or a text string)
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)

    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = util.semantic_search(query_emb, embeddings, top_k=k)[0]
    return [image_ids[hit["corpus_id"]] for hit in hits]


def tag_search_results(syno: SynoPhotosSharedAPI, tag_id: int, image_ids_to_tag: list):
    return syno.add_tags(image_ids=image_ids_to_tag, tag_ids=tag_id)


def tag_subset(syno: SynoPhotosSharedAPI, tags_created: dict, tag_name: str, k: int):
    # images are provided by search in reverse sorted order by score
    image_ids_to_remove = tags_created[tag_name]["image_ids"][k:]
    if image_ids_to_remove:
        syno.remove_tags(
            image_ids=image_ids_to_remove,
            tag_ids=tags_created[tag_name]["id"],
        )


def delete_tags(syno: SynoPhotosSharedAPI, tags: dict[str, int]):
    for tag_name in tags:
        logger.info(f"Removing tag `{tag_name}` from all associated images...")
        syno.remove_tags(
            image_ids=tags[tag_name]["image_ids"],
            tag_ids=tags[tag_name]["id"],
        )
        logger.info(
            f"Tag `{tag_name}` has been removed from all {len(tags[tag_name]['image_ids'])} images."
        )


def user_input_select_and_delete_tags(syno: SynoPhotosSharedAPI, tags_created: dict):
    tags_to_delete = {}
    for tag in tags_created:
        option = utils.get_user_choice(
            choices=["yes", "no"], prompt=f"Would you like to delete tag - `{tag}`?"
        )
        if option == "yes":
            tags_to_delete[tag] = tags_created[tag]
    delete_tags(syno, tags_to_delete)


def user_input_collect_tag_names():
    tag_names_to_delete = []
    while True:
        tag_name = input(f"Enter tag name to delete. Leave empty to exit : ")
        if not tag_name:
            logger.info(
                "Collected tag names to delete. Only tags matching those in Synology Photos will be deleted."
            )
            break
        tag_names_to_delete.append(tag_name)
    return tag_names_to_delete


def user_input_collect_and_delete_tags(syno: SynoPhotosSharedAPI):
    tags_to_delete = None
    option = utils.get_user_choice(
        choices=["manual", "list"],
        prompt="Would you like to manually enter tags to delete, or choose from list of each tag in Synology Photos?",
    )
    if option == "manual":
        tag_names_to_delete = user_input_collect_tag_names()
        if tag_names_to_delete:
            tags_to_delete = create_tag_image_dict(syno, tag_names=tag_names_to_delete)
            delete_tags(syno, tags_to_delete)
    elif option == "list":
        all_tags_dict = create_tag_image_dict(syno, tag_names=None)
        user_input_select_and_delete_tags(syno, all_tags_dict)


def user_input_delete_tags(syno: SynoPhotosSharedAPI, tags_created: dict):
    if not tags_created:
        return
    option = utils.get_user_choice(
        choices=["all", "none", "select"],
        prompt="Would you like to delete tags created in this session?",
    )
    if option == "all":
        delete_tags(syno, tags_created)
    elif option == "select":
        user_input_select_and_delete_tags(syno, tags_created)


def main():
    parser = argparse.ArgumentParser(
        description="inputs for searching in Synology Photos using NLP, and tagging images in Synology with the results"
    )
    parser.add_argument(
        "--emb_file",
        type=str,
        help="relative or absolute path of the embeddings file",
    )
    parser.add_argument(
        "--syno_host",
        type=str,
        help="ip or hostname of Synology server",
    )
    parser.add_argument(
        "--emb_batch",
        required=False,
        type=int,
        default=16,
        help="number of images to embed at a time. default=16",
    )
    parser.add_argument(
        "--nn_batch",
        required=False,
        type=int,
        default=4,
        help="batch size used in the transformer model. default=4",
    )
    parser.add_argument(
        "--torch_threads",
        required=False,
        type=int,
        default=1,
        help="number of threads used by torch. must be >0. default=1",
    )
    parser.add_argument(
        "--max_images",
        required=False,
        type=int,
        help="maximum number of images to embed. Not providing this embeds all images",
    )
    parser.add_argument(
        "--log_file",
        required=False,
        type=str,
        help="relative or absolute path of the logs file. not providing this only logs to stdout",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        type=int,
        default=1,
        help="verbose level for the log file. 0 = warning, 1 = info, 2 = debug. default=1",
    )
    # TODO - change no_reindex to update_embeddings
    parser.add_argument(
        "--no_index",
        required=False,
        default=False,
        action="store_true",
        help="skip indexing to create embeddings for any recent images before searching",
    )
    parser.add_argument(
        "--update_embeddings",
        required=False,
        default=False,
        action="store_true",
        help="only update embeddings, skipping all other tasks",
    )
    parser.add_argument(
        "--delete_tags",
        required=False,
        default=False,
        action="store_true",
        help="only delete tags, skipping all other tasks",
    )

    args = parser.parse_args()

    emb_file = args.emb_file
    hostname = args.syno_host
    emb_batch = args.emb_batch
    nn_batch = args.nn_batch
    torch_threads = args.torch_threads
    max_images = args.max_images
    log_file = args.log_file
    verbose = args.verbose
    no_index = args.no_index
    update_embeddings_only = args.update_embeddings
    delete_tags_only = args.delete_tags

    utils.configure_logger(
        logger, utils.get_absolute_path_in_container(log_file), verbose=verbose
    )
    utils.configure_logger(
        logging.getLogger("utils"),
        utils.get_absolute_path_in_container(log_file),
        verbose=verbose,
    )
    utils.configure_logger(
        logging.getLogger("syno_generate_embeddings"),
        utils.get_absolute_path_in_container(log_file),
        verbose=verbose,
    )
    utils.configure_logger(
        logging.getLogger("synoapi"),
        utils.get_absolute_path_in_container(log_file),
        verbose=verbose,
    )

    logger.debug(f"Arguments passed to {__name__} - {args}")

    # login into Synology Photos
    syno = SynoPhotosSharedAPI(hostname=hostname)
    syno.login(password_trials=5)

    tags_created = dict()
    try:
        if delete_tags_only:
            user_input_collect_and_delete_tags(syno)
            return

        # create image path tree on synology
        image_id_path_dict = create_image_path_tree(syno)

        # load in model to embed search term
        logger.info(f"Loading ML embeddings model files...")
        model = load_model()
        logger.info(f"Loaded ML embeddings model files.")

        # ask user to update embeddings
        if update_embeddings_only or (
            (not no_index) and get_user_approval(request="update_embeddings")
        ):
            update_embeddings(
                model,
                utils.get_absolute_path_in_container(emb_file),
                image_id_path_dict,
                torch_threads,
                emb_batch,
                nn_batch,
                max_images,
            )
        if update_embeddings_only:
            return

        # read in embeddings pkl file
        embedded_images = utils.read_emb_file(
            utils.get_absolute_path_in_container(emb_file)
        )
        if not embedded_images:
            message = f"There are no embedded images in `{emb_file}`"
            logger.error(message)
            raise ValueError(message)
        image_ids, embeddings = embedded_images["ids"], embedded_images["embeddings"]

        # start search
        if not get_user_approval(request="search_embeddings"):
            return
        while True:
            # get search term from user
            query, k = get_user_query()
            if not query:
                logger.info("Exiting search program.")
                break
            tag_name = f"nlp_search {query}"
            # tag images that match search term
            if tag_name in tags_created and len(tags_created[tag_name]["results"]) >= k:
                tag_subset(syno, tags_created, tag_name, k)
            else:
                results = search(query, model, image_ids, embeddings, k=k)
                tag_dict = syno.create_tag(tag_name)
                tag_id = tag_dict["id"]
                tag_search_results(syno, tag_id, results)
                tags_created[tag_name] = dict(image_ids=results, id=tag_id)

            logger.info(f"Tagged {k} images for `{query}`.")
            print(f"Filter for tag `{tag_name}` in Synology Photos.")
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        user_input_delete_tags(syno, tags_created)
        # logout
        syno.logout()


if __name__ == "__main__":
    main()
