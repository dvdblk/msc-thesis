import argparse
import time
import os
import sys
import traceback

import torch
import torch.multiprocessing as mp

import structlog
from qdrant_client import models
from tqdm import tqdm

from queue import Empty as QueueEmptyException
from app.utils import configure_structlog, setup_qdrant_client
from app.models import setup_model_and_tokenizer, get_model_names_list
from app.explainers import (
    get_xai_method_names_list,
    process_abstract,
    get_explainer,
    ExplainerMethod,
)

configure_structlog()
log = structlog.get_logger()


def worker_function(task_queue, result_queue, model, tokenizer, args):
    worker_id = torch.multiprocessing.current_process()._identity[0] - 1

    torch.cuda.init()
    visible_devices_str = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_devices_str is not None:
        visible_devices = [int(x) for x in visible_devices_str.split(",")]
    else:
        visible_devices = list(range(torch.cuda.device_count()))

    num_gpus = len(visible_devices)
    gpu_id = worker_id % num_gpus
    real_gpu_id = visible_devices[gpu_id]
    device = torch.device(f"cuda:{gpu_id}")
    log = structlog.get_logger().bind(worker_id=worker_id, real_gpu_id=real_gpu_id)
    model = model.to(device)
    explainer = get_explainer(args.method, model, tokenizer, device, args)
    log.info("Setting up worker", device=device)

    while True:
        publications = task_queue.get()
        if publications is None:  # This is our signal that no more data is coming
            break
        log.debug("Got publications", n_publications=len(publications))

        for publication in publications:
            abstract = publication.payload.get("description", "")
            title = publication.payload.get("title", "")
            publication_id = publication.payload.get("id", "")
            existing_xai = publication.payload.get("xai", [])
            point_id = publication.id

            preprocessed_abstract = title + " " + abstract
            try:
                predicted_label, probs, xai_values = process_abstract(
                    preprocessed_abstract, explainer, model, tokenizer, device
                )

                # TODO: add postprocessing here e.g. for tokens etc
                if ExplainerMethod(args.method) == ExplainerMethod.ATTNLRP:
                    # replace "[CLS]" and "[SEP]" tokens with empty string
                    if xai_values["input_tokens"][0] == "[CLS]":
                        xai_values["input_tokens"][0] = ""
                    if xai_values["input_tokens"][-1] == "[SEP]":
                        xai_values["input_tokens"][-1] = ""

                existing_xai.append(
                    {
                        "xai_method": args.method,
                        "model_family": args.model_family,
                        "model_path": args.model_path,
                        "predicted_label": predicted_label,
                        "probs": probs,
                        "xai_values": xai_values,
                    }
                )
                result_queue.put((publication_id, point_id, existing_xai))
                log.debug("Processed abstract", publication_id=publication_id)
            except Exception as e:
                log.error(f"Error processing abstract {publication_id}: {str(e)}")
                log.error(traceback.format_exc())
                result_queue.put((publication_id, point_id, None))


def main(args):

    model, tokenizer = setup_model_and_tokenizer(args)
    log.info("Loaded model and tokenizer")

    # Connect to Qdrant
    client = setup_qdrant_client()
    log.info("Qdrant connection set up.")

    # Set up multiprocessing
    num_gpus = torch.cuda.device_count()
    num_workers = num_gpus * args.n_workers  # workers per gpu
    # Note: takes a while because of "spawn", but it is required to share the same model across processes
    mp.set_start_method("spawn", force=True)

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    with mp.Pool(
        num_workers,
        initializer=worker_function,
        initargs=(task_queue, result_queue, model, tokenizer, args),
    ):

        # Process publications
        offset = 0
        total_processed = 0
        total_queued = 0
        batch_size = 500
        n_publications = args.n_publications
        queue_low_water_mark = num_workers
        queue_high_water_mark = num_workers * 2

        def queue_publications():
            nonlocal offset, total_queued
            while task_queue.qsize() * batch_size < queue_high_water_mark:
                publications = client.scroll(
                    collection_name="publications",
                    limit=batch_size,
                    offset=offset,
                    # skips already processed publications with this method, model family and path
                    scroll_filter=models.Filter(
                        must_not=[
                            models.FieldCondition(
                                key="xai[].xai_method",
                                match=models.MatchValue(value=args.method),
                            ),
                            models.FieldCondition(
                                key="xai[].model_family",
                                match=models.MatchValue(value=args.model_family),
                            ),
                            models.FieldCondition(
                                key="xai[].model_path",
                                match=models.MatchValue(value=args.model_path),
                            ),
                        ]
                    ),
                    with_payload=["description", "id", "title", "xai"],
                )
                if not publications[0]:
                    return False
                n_pubs = len(publications[0])
                task_queue.put(publications[0])
                offset += n_pubs
                total_queued += n_pubs
                log.debug(
                    "Queued more publications",
                    n_added=n_pubs,
                    total_queued=total_queued,
                )
            return True

        def queue_publications_with_label():
            nonlocal offset, total_queued
            chunk_size = 20

            while task_queue.qsize() * batch_size < queue_high_water_mark:
                publications = client.scroll(
                    collection_name="publications",
                    limit=batch_size,
                    offset=offset,
                    # skips already processed publications with this method, model family and path
                    scroll_filter=models.Filter(
                        must_not=[
                            models.IsNullCondition(
                                is_null=models.PayloadField(key="labels")
                            )
                        ]
                    ),
                    with_payload=["description", "id", "title", "xai"],
                )
                if not publications[0]:
                    return False

                for i in range(0, len(publications[0]), chunk_size):
                    chunk = publications[0][i : i + chunk_size]
                    task_queue.put(chunk)

                n_pubs = len(publications[0])
                offset += n_pubs
                total_queued += n_pubs
                log.debug(
                    "Queued more publications",
                    n_added=n_pubs,
                    total_queued=total_queued,
                )
            return True

        log.info("Starting to process abstracts")
        # sleep for a short while to avoid tqdm output bug
        time.sleep(5)
        queue_publications_with_label()

        with tqdm(
            total=n_publications, file=sys.stdout, position=0, leave=True
        ) as pbar:
            while total_processed < n_publications and (
                total_processed < total_queued or total_queued < n_publications
            ):
                # # Check if we need to queue more publications
                # if task_queue.qsize() * batch_size <= queue_low_water_mark:
                #     if not queue_publications_with_label():
                #         # No more publications to queue
                #         if task_queue.empty() and total_processed > 0:
                #             break

                try:
                    result = result_queue.get(timeout=60)
                    if result is not None:
                        publication_id, point_id, new_xai_field = result
                        if new_xai_field is not None:
                            log.debug(
                                "Created XAI for abstract",
                                publication_id=publication_id,
                                point_id=point_id,
                            )
                            client.set_payload(
                                collection_name="publications",
                                points=[point_id],
                                payload={
                                    "xai": new_xai_field,
                                },
                            )
                        total_processed += 1

                        # Update tqdm
                        pbar.set_postfix(in_queue=total_queued - total_processed)
                        pbar.update(1)

                        if total_processed % 100 == 0:
                            log.info(
                                f"Progress update",
                                total_processed=total_processed,
                                total_queued=total_queued,
                            )
                except QueueEmptyException:
                    log.warning(
                        f"Result queue is empty (timed out after 10s). Waiting for more results..."
                    )

        # Signal workers to stop
        for _ in range(num_workers):
            task_queue.put(None)

        log.info(f"Finished processing abstracts", total_processed=total_processed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="SDG text-classification explainer CLI",
        description="Loads a model and explains its predictions on given abstracts.",
    )
    parser.add_argument(
        "--method",
        help="Explainer method to use e.g. 'shap' for parition shap.",
        choices=get_xai_method_names_list(),
        required=True,
    )
    parser.add_argument(
        "--model-family",
        help="The model family to use.",
        choices=get_model_names_list(),
        required=True,
    )
    parser.add_argument(
        "--model-path",
        help="Checkpoint path to the model to explain, otherwise uses the default model of the family.",
        required=True,
    )
    parser.add_argument(
        "--n-workers",
        help="Number of workers PER GPU to use for parallel processing (computation of XAI values).",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n-publications",
        help="Limit of publications to process in total.",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--tfidf-corpus-path",
        help="Path to the corpus for TF-IDF vectorizer in SHAP Partition TF-IDF",
        required=False,
    )

    # FIXME: Add publication retrieval method (e.g. qdrant vs dataset)
    # parser.add_argument("--pub-retrieval-method", help="Method to retrieve publications.")
    parser.add_argument("-o", "--output-path", help="Path to save the explanations.")
    args = parser.parse_args()

    # Validate args
    if args.method == "shap-partition-tfidf" and not args.tfidf_corpus_path:
        parser.error(
            "The --tfidf-corpus-path argument is required for the 'shap-partition-tfidf' method"
        )

    log.info(f"Starting explainer", method=args.method, model_family=args.model_family)
    main(args)
