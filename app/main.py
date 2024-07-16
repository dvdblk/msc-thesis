import argparse
import time
import os
import sys
import traceback

import torch
import torch.multiprocessing as mp

import structlog
from dotenv import load_dotenv
import shap
from shap import PartitionExplainer, Explainer
from shap.maskers import Text as TextMasker
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
import logging
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from queue import Empty as QueueEmptyException

load_dotenv()


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def configure_structlog():
    structlog.configure(
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s", level=logging.INFO, handlers=[TqdmLoggingHandler()]
    )
    # ignore INFO messages from httpx / qdrant
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("shap").setLevel(logging.WARNING)


configure_structlog()
log = structlog.get_logger()

__methods_map = {"shap": PartitionExplainer, "tfidf-shap": None, "attnlrp": None}
__model_family_map = {
    "llama": None,
    "unllama": None,  # unmasked llama
    "scibert": None,
}


def setup_model_and_tokenizer(args):
    # FIXME: move somewhere
    id2label = {i: str(i + 1) for i in range(17)}
    label2id = {str(i + 1): i for i in range(17)}

    # Load model
    # FIXME: add model loading logic based on args
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path or __model_family_map[args.model_family],
        num_labels=17,
        id2label=id2label,
        label2id=label2id,
    )
    model.eval()
    model.share_memory()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    return model, tokenizer


def process_abstract(abstract, model, tokenizer, device):
    def predictor(texts):
        inputs = tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.cpu()

        # softmax
        probs = torch.exp(logits) / torch.exp(logits).sum(-1, keepdims=True)
        return probs

    # get partition shap explanations
    explainer = Explainer(
        predictor, algorithm="partition", masker=TextMasker(tokenizer), silent=True
    )
    shap_values = explainer([abstract])

    # get prediction
    # (we can also get this by summing base_values with shap_values but this is more explicit)
    inputs = tokenizer(
        [abstract], padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # get the predicted id
    predicted_id = torch.argmax(probs, dim=-1).item()
    prediction = torch.argmax(outputs.logits, dim=1).item()
    assert predicted_id == prediction

    # reorder probs to match the order of the labels
    sorted_labels = sorted(model.config.id2label.items(), key=lambda x: int(x[1]))
    # create permutation tensor based on label order
    probs_permutation = torch.tensor([label_index for label_index, _ in sorted_labels])
    probs = probs.squeeze()[probs_permutation]
    # need to add +1 because the labels are 0-indexed
    predicted_label = torch.argmax(probs).item() + 1

    # apply probs_permutation on shap_values in the last dimension

    log.info(shap_values)
    log.info(shap_values.shape)
    exit()

    # return (prediction, xai_values)
    xai_values = {
        "token_scores": shap_values.values[0].tolist(),
        # "base_values": shap_values.base_values[0].tolist(),
        "input_tokens": shap_values.data[0].tolist(),
    }
    assert len(xai_values["token_scores"]) == len(xai_values["input_tokens"])

    return predicted_label, probs.tolist(), xai_values


def worker_function(task_queue, result_queue, model, tokenizer, args):
    worker_id = torch.multiprocessing.current_process()._identity[0] - 1
    visible_devices_str = os.getenv(
        "CUDA_VISIBLE_DEVICES", ",".join(map(str, range(torch.cuda.device_count())))
    )
    visible_devices = [int(x) for x in visible_devices_str.split(",")]

    num_gpus = len(visible_devices)
    gpu_id = worker_id % num_gpus
    device = torch.device(f"cuda:{visible_devices[gpu_id]}")
    log = structlog.get_logger().bind(worker_id=worker_id, gpu_id=gpu_id)
    model = model.to(device)
    log.info("Setting up worker", device=device)

    while True:
        publications = task_queue.get()
        if publications is None:  # This is our signal that no more data is coming
            break

        for publication in publications:
            abstract = publication.payload.get("description", "")
            title = publication.payload.get("title", "")
            publication_id = publication.payload.get("id", "")
            existing_xai = publication.payload.get("xai", [])
            point_id = publication.id

            preprocessed_abstract = title + " " + abstract
            try:
                result = process_abstract(
                    preprocessed_abstract, model, tokenizer, device
                )
                existing_xai.append(
                    {
                        "xai_method": args.method,
                        "model_family": args.model_family,
                        "model_path": args.model_path,
                        "predicted_label": result[0],
                        "probs": result[1],
                        "xai_values": result[2],
                    }
                )
                result_queue.put((publication_id, point_id, existing_xai))
            except Exception as e:
                log.error(f"Error processing abstract {publication_id}: {str(e)}")
                log.error(traceback.format_exc())
                result_queue.put((publication_id, point_id, None))


def main(args):

    model, tokenizer = setup_model_and_tokenizer(args)

    # Connect to Qdrant
    client = QdrantClient(
        os.getenv("QDRANT_API_URL"),
        port=os.getenv("QDRANT_API_PORT"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    log.info("Loaded model, tokenizer and connected to Qdrant")

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
        batch_size = 4
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
                            ),
                            # models.FieldCondition(
                            #     key="xai[].xai_method",
                            #     match=models.MatchValue(value=args.method),
                            # ),
                            # models.FieldCondition(
                            #     key="xai[].model_family",
                            #     match=models.MatchValue(value=args.model_family),
                            # ),
                            # models.FieldCondition(
                            #     key="xai[].model_path",
                            #     match=models.MatchValue(value=args.model_path),
                            # ),
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

        log.info("Starting to process abstracts")
        # sleep for a short while to avoid tqdm output bug
        time.sleep(1)
        queue_publications_with_label()

        with tqdm(total=n_publications, file=sys.stdout) as pbar:
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
                    result = result_queue.get(timeout=10)
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
        choices=__methods_map.keys(),
        required=True,
    )
    parser.add_argument(
        "--model-family",
        help="The model family to use.",
        choices=__model_family_map.keys(),
        required=True,
    )
    parser.add_argument(
        "--model-path",
        help="Checkpoint path to the model to explain, otherwise uses the default model of the family.",
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
    # FIXME: Add publication retrieval method (e.g. qdrant vs dataset)
    # parser.add_argument("--pub-retrieval-method", help="Method to retrieve publications.")
    parser.add_argument("-o", "--output-path", help="Path to save the explanations.")
    args = parser.parse_args()

    log.info(f"Starting explainer", method=args.method, model_family=args.model_family)
    main(args)
