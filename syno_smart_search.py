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


def create_image_path_tree(syno: SynoPhotosSharedAPI):
    # get files with their filepaths as dict{filepath:id}
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
    return result


def get_user_approval(request: str):
    if request == "update_embeddings":
        result = input(
            "(time consuming step) Would you like to update image index to include recent images in search? (\033[4my\033[0mes | \033[4mn\033[0mo) : "
        )
    elif request == "search_embeddings":
        result = input(
            "Would you like to search through images? (\033[4my\033[0mes | \033[4mn\033[0mo) : "
        )
    else:
        message = f"'{request}' is not a valid request option."
        logger.error(message)
        raise ValueError(message)
    while True:
        if result.lower() == "yes" or result.lower() == "y":
            return True
        elif result.lower() == "no" or result.lower() == "n":
            return False
        else:
            result = input(
                "Please select a valid option (\033[4my\033[0mes | \033[4mn\033[0mo) : "
            )


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
    image_ids_to_remove = tags_created[tag_name]["results"][k:]
    if image_ids_to_remove:
        syno.remove_tags(
            image_ids=image_ids_to_remove,
            tag_ids=tags_created[tag_name]["id"],
        )


def delete_tags(syno: SynoPhotosSharedAPI, tags_created: dict):
    for tag_name in tags_created:
        syno.remove_tags(
            image_ids=tags_created[tag_name]["results"],
            tag_ids=tags_created[tag_name]["id"],
        )
        logger.info(f"Tag '{tag_name}' has been deleted.")


def user_input_delete_tags(syno: SynoPhotosSharedAPI, tags_created: dict):
    if not tags_created:
        return
    option = input(
        "Would you like to delete tags created in this session? (\033[4ma\033[0mll | \033[4ms\033[0melect | \033[4mn\033[0mone) : "
    )
    while True:
        if option.lower() == "all" or option.lower() == "a":
            delete_tags(syno, tags_created)
            return
        if option.lower() == "none" or option.lower() == "n":
            return
        if option.lower() == "select" or option.lower() == "s":
            tags_to_delete = {}
            for tag in tags_created:
                option2 = input(
                    f"Would you like to delete tag - {tag} (\033[4my\033[0mes | \033[4mn\033[0mo) : "
                )
                while True:
                    if option2.lower() == "yes" or option2.lower() == "y":
                        tags_to_delete[tag] = tags_created[tag]
                        break
                    elif option2.lower() == "no" or option2.lower() == "n":
                        break
                    else:
                        option2 = input(
                            "Please select a valid option (\033[4my\033[0mes | \033[4mn\033[0mo) : "
                        )
            delete_tags(syno, tags_to_delete)
            return
        option = input(
            "Please select a valid option (\033[4ma\033[0mll | \033[4ms\033[0melect | \033[4mn\033[0mone) : "
        )


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
        help="url or ip of synology device",
    )
    parser.add_argument(
        "--emb_batch",
        required=False,
        type=int,
        default=1024,
        help="number of images to embed at a time. default=1024",
    )
    parser.add_argument(
        "--nn_batch",
        required=False,
        type=int,
        default=16,
        help="batch size used in transformer model. default=16",
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
        help="verbose level for the log file. 0 = warning, 1 = info, 2 = debug",
    )
    parser.add_argument(
        "--no_reindex",
        required=False,
        default=False,
        action="store_true",
        help="skip reindexing to create embeddings for any recent images before searching",
    )
    parser.add_argument(
        "--no_search",
        required=False,
        default=False,
        action="store_true",
        help="skip searching workflow. useful to only perform reindexing",
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
    no_reindex = args.no_reindex
    no_search = args.no_search

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
    syno.login(trials=5)

    tags_created = dict()
    try:
        # create image path tree on synology
        image_id_path_dict = create_image_path_tree(syno)

        # load in model to embed search term
        model = load_model()

        # ask user to update embeddings
        if (not no_reindex) and get_user_approval(request="update_embeddings"):
            update_embeddings(
                model,
                utils.get_absolute_path_in_container(emb_file),
                image_id_path_dict,
                torch_threads,
                emb_batch,
                nn_batch,
                max_images,
            )

        # read in embeddings pkl file
        embedded_images = utils.read_emb_file(
            utils.get_absolute_path_in_container(emb_file)
        )
        if not embedded_images:
            message = f"There are no embedded images in {emb_file}"
            logger.error(message)
            raise ValueError(message)
        image_ids, embeddings = embedded_images["ids"], embedded_images["embeddings"]

        # start search
        perform_search = True
        if no_search or (not get_user_approval(request="search_embeddings")):
            perform_search = False
        while True:
            if not perform_search:
                break
            # get search term from user
            query, k = get_user_query()
            if query is None:
                logger.info("Exiting search program.")
                break
            tag_name = f"nlp_api {query}"
            # tag images that match search term
            if tag_name in tags_created and len(tags_created[tag_name]["results"]) >= k:
                tag_subset(syno, tags_created, tag_name, k)
            else:
                results = search(query, model, image_ids, embeddings, k=k)
                tag_dict = syno.create_tag(tag_name)
                tag_id = tag_dict["id"]
                tag_search_results(syno, tag_id, results)
                tags_created[tag_name] = dict(results=results, id=tag_id)

            logger.info(f"Tagged {k} images for '{query}'.")
            print(f"Search for tag '{tag_name}' in Synology Photos.")
    except Exception as e:
        logger.error(e)
    finally:
        user_input_delete_tags(syno, tags_created)
        # logout
        syno.logout()


if __name__ == "__main__":
    main()
