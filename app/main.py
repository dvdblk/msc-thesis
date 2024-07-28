import argparse
import time
import os
import sys
import traceback

import torch
import torch.multiprocessing as mp

import structlog
from tqdm import tqdm

from queue import Empty as QueueEmptyException
from app.utils import configure_structlog
from app.utils.enum_action import EnumAction
from app.models import setup_model_and_tokenizer, get_model_names_list
from app.explainers import (
    get_xai_method_names_list,
    get_explainer,
)
from app.data.enum import DataSource
from app.explainers.model import XAIOutput
from app.data.manager import QdrantManager, LocalManager

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
            preprocessed_abstract = ""
            if publication.title is not None:
                preprocessed_abstract += publication.title + " "
            preprocessed_abstract += publication.abstract

            try:
                # Get the explanation
                xai_output: XAIOutput = explainer.explain(preprocessed_abstract)
                result_queue.put((publication, xai_output))
                log.debug("Processed abstract", publication_id=publication.zora_id)

            except Exception as e:
                log.error(f"Error processing abstract {publication.zora_id}: {str(e)}")
                log.error(traceback.format_exc())
                result_queue.put((publication, None))


def main(args):

    model, tokenizer = setup_model_and_tokenizer(args)
    log.info("Loaded model and tokenizer")

    # Setup data manager
    data_manager = None
    if args.data_source_target == DataSource.QDRANT:
        data_manager = QdrantManager(
            "publications", load_only_labelled_data=True, batch_size=args.batch_size
        )
    elif args.data_source_target == DataSource.LOCAL:
        data_manager = LocalManager(
            "./experiments/data/zo_up.csv",
            "./explanations.json",
            batch_size=args.batch_size,
        )
    # log.info("Data manager set up", data_source_and_target=args.data_source_target)

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
        total_processed = 0
        total_queued = 0
        n_publications = args.n_publications
        queue_low_water_mark = num_workers
        queue_high_water_mark = num_workers * 2

        def queue_publications(chunk_size=20):
            nonlocal total_queued
            while task_queue.qsize() * args.batch_size < queue_high_water_mark:
                publications = data_manager.load_data()
                if publications is None:
                    return False
                for i in range(0, len(publications), chunk_size):
                    chunk = publications[i : i + chunk_size]
                    task_queue.put(chunk)

                n_pubs = len(publications)
                total_queued += n_pubs
                log.debug(
                    "Queued more publications",
                    n_added=n_pubs,
                    total_queued=total_queued,
                )
            return True

        log.info("Starting to process abstracts")
        # sleep for a short while to avoid tqdm output bug
        time.sleep(10)
        queue_publications()

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
                        publication, xai_output = result
                        if xai_output is not None:
                            data_manager.store_data(
                                publication=publication,
                                xai_output=xai_output,
                                model_family=args.model_family,
                                model_path=args.model_path,
                            )
                            log.debug(
                                "Created XAI for abstract",
                                id=publication.zora_id,
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

        time.sleep(5)
        log.info(f"Finished processing abstracts", total_processed=total_processed)

    # Close data manager
    data_manager.close()

    log.info("Exiting...")

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
        "--batch-size",
        help="Batch size for loading data.",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--tfidf-corpus-path",
        help="Path to the corpus for TF-IDF vectorizer in SHAP Partition TF-IDF",
        required=False,
    )
    # TODO: maybe allow for multiple data sources (e.g. load local file, upload to qdrant etc)
    parser.add_argument(
        "--data-source-target",
        type=DataSource,
        action=EnumAction,
        required=True,
        help="Specifies the source and target of input data (abstracts, explanations)",
    )
    # parser.add_argument(
    #     "--output-dest",
    #     type=OutputDestination,
    #     choices=list(OutputDestination),
    #     required=True,
    #     help="Specifies the destination for output data (explanations)",
    # )

    parser.add_argument("-o", "--output-path", help="Path to save the explanations.")
    args = parser.parse_args()

    # Validate args
    if args.method == "shap-partition-tfidf" and not args.tfidf_corpus_path:
        parser.error(
            "The --tfidf-corpus-path argument is required for the 'shap-partition-tfidf' method"
        )

    log.info(
        f"Starting explainer",
        data_source_target=args.data_source_target.value,
        method=args.method,
        model_family=args.model_family,
    )
    main(args)
