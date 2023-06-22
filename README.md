# synology-photos-nlp-search
Search through Synology Photos using natural language captions

## Instructions
1. Mount the photos directory on Synology network drive to your local machine.
1. Install and open [docker](https://www.docker.com/products/docker-desktop/).
1. Provide argument values in `env` file. Instructions are in the file. Save the file as `.env`.
1. Start the search through image embeddings using - 
    ```
    docker-compose run --build --rm -i synosearch
    ```
    Follow the instructions when the program runs for indecing images (i.e., creating image embeddings) and performing searches.
    
    If running for the first time, embeddings will be created for all images in Synology Photos. This can be a multi-day process depending on available compute resources.
1. After completing the search, optionally turn down docker-compose using - 
    ```
    docker-compose down -v
    ```
1. There are optional parameters that can be used when starting the search - 
    1. To skip generation of embeddings of any recent images and directly search, `--no_reindex` flag can be used - 
        ```
        docker-compose run --build --rm synosearch --no_reindex
    1. Similarly, to skip search when only wanting to generate embeddings, `--no_search` flag can be used.
    1. To only delete tags, run the command with `--delete_tags` flag.

## How It Works
1. Each image is converted to its vector embedding using a bi-modal embeddings ML model. Generating embeddings for all images can be a time-consuming process based on compute power of the local machine and the number of images. 
1. When a search is performed, the search term is converted to a vector embedding using the same model. 
1. The most similar images to the search term, based on their embeddings, are tagges using Synology Photos API.
1. The Synology Photos App can then be used to browse the images based on the tag created for the search term.
1. Finally, there is an option to delete the tags that were created in Synology Photos for the search term.