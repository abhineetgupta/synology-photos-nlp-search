synology-photos-nlp-search
======
Search through Synology Photos in the Shared space using natural language captions, e.g., _sunset on a beach_, _dog catching a frisbee_, etc. 

The setup is containerized and designed to run on a host machine separate from the Synology server. Prior to the first run, update the environment variables that will be used to connect with the Synology server. 

The first run would require indexing (i.e., generating embeddings for all images in Synology Photos) which is a time-intensive process dependent on the number of images and the available compute on the host machine. It is recommended to let the script run in the background until all images are embedded. This can take a few hours to days but progress is saved regularly. In subsequent runs, only new and deleted images are updated and indexing will be faster.

## Instructions
1. Mount the photos directory from Synology on the host machine. The directory should be the same that is used for the Shared space in Synology Photos.
1. Clone this repo.
1. Provide argument values in `env` file. Instructions are in the file. Save the file as `.env`.
1. Open Docker. Install from [docker.com](https://www.docker.com/products/docker-desktop/) if not available.
1. Open `terminal` and `cd` into the cloned repo.
1. Start the search through image embeddings using - 
    ```
    docker-compose run --build --rm -i synosearch
    ```
    The user is prompted for whether to index images (i.e., create embeddings), and for search terms. Tags are created for each search term to be used in the Synology Photos GUI to view search results. After the user is done searching, there is a prompt for whether to delete tags that were created in the session.
    
    *Note*: If running for the first time, embeddings will be created for all images in Shared space of Synology Photos. This can be a multi-hours to -days process depending on Docker's compute resources on the host and the number of images.
1. There are optional parameters that can be used to change the script's behavior - 
    - To skip indexing any recent images and instead go the the search step directly, `--no_index` flag can be used - 
        ```
        docker-compose run --build --rm -i synosearch --no_index
        ```
    - Similarly, to skip search and only generate embeddings, `--update_embeddings` flag can be used.
    - To remove existing tags from images (and skip both image indexing and search), use the `--delete_tags` flag.

## How It Works
1. Each image is converted to its vector embedding using a bi-modal transformers-based embeddings ML model - [clip-ViT-B-32](https://huggingface.co/sentence-transformers/clip-ViT-B-32). Generating embeddings for all images can be a time-consuming process based on compute power of the host machine and the number of images. 
1. When a search is performed, the search term is converted to a vector embedding using the same model. 
1. The most similar images to the search term, based on their embeddings, are tagged using the Synology Photos API.
1. The Synology Photos GUI can then be used to view images with the tag created for each search term.

## Limitations
1. Synology Photos has a separate API for Personal and Shared spaces. The current API is implemented and tested for Shared space only.
1. There is no official documentation for Synology Photos API and most of the API implementation is based on trial-and-error and other similar projects like [SynologyPhotosAPI](https://github.com/zeichensatz/SynologyPhotosAPI).
1. Synology Photos API calls can be slow to process and error out without a lot of information. The script auto-tries a few times before completely exiting the program.
1. Only `pytorch-cpu` is implemented and tested.

## To Do
1. Update the `Dockerfile` and embeddings generation steps for `pytorch` to use GPU, if available.