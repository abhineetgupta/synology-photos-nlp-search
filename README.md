# synology-photos-nlp-search
Search through Synology Photos using natural language captions

## Instructions
1. Install and open [docker](https://www.docker.com/products/docker-desktop/).
1. Provide argument values in `env` file. Instructions are in the file. Save the file as `.env`.
1. Start the search through image embeddings using - 
    ```
    docker-compose run --build synosearch
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
        docker-compose run --build synosearch --no_reindex
    1. Similarly, to skip search when only wanting to generate embeddings, `--no_search` flag can be used.