import argparse
import time
import os
import sys
import traceback

import torch
import torch.multiprocessing as mp
import numpy as np

import structlog
from tqdm import tqdm

from queue import Empty as QueueEmptyException
from lxt.utils import pdf_heatmap, clean_tokens
from app.utils import configure_structlog
from app.utils.enum_action import EnumAction
from app.utils.tokenization import (
    prepare_fixed_bert_tokens_for_pdf_viz,
    truncate_to_bert_limit,
)
from app.models import setup_model_and_tokenizer, get_model_names_list
from app.explainers import (
    get_xai_method_names_list,
    get_explainer,
)
from app.data.enum import DataSource
from app.explainers.model import XAIOutput, ExplainerMethod
from app.evaluate import XAIEvaluator
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


def explain(args, model, tokenizer, data_manager):

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
        time.sleep(2)
        queue_publications()

        # in seconds
        queue_waiting_timeout = 120
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
                    result = result_queue.get(timeout=queue_waiting_timeout)
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
                        f"Result queue is empty (timed out after {queue_waiting_timeout}s). This can happen if explanation methods are taking too long to generate explanations. Waiting for more results..."
                    )

        # Signal workers to stop
        for _ in range(num_workers):
            task_queue.put(None)

        time.sleep(5)
        log.info(f"Finished processing abstracts", total_processed=total_processed)

    # Close data manager
    data_manager.close()

    log.info("Exiting...")

def visualize(args, model, tokenizer, data_manager):
    # Load data into a DataFrame
    df = data_manager.load_data()

    # Get one gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log = structlog.get_logger().bind(device=device)
    model = model.to(device)
    explainer = get_explainer(args.method, model, tokenizer, device, args)

    # Get abstract by index and explanation
    abstract = df.iloc[args.index].abstract
    abstract = truncate_to_bert_limit(abstract, tokenizer)

    # Produce explanation
    xai_output: XAIOutput = explainer.explain(abstract)

    # Turn tokens scores into a tensor and transpose
    # token_scores is now of shape (num_classes, num_tokens)
    token_scores = torch.tensor(xai_output.token_scores).transpose(0, 1)

    # Clamp scores in the range [-1, 1] for some methods
    if (
        xai_output.xai_method == ExplainerMethod.INTEGRATED_GRADIENT
        or xai_output.xai_method == ExplainerMethod.GRADIENTXINPUT
    ):
        token_scores = torch.clamp(token_scores, min=-1.0, max=1.0)

    class_index = args.class_index or xai_output.predicted_id

    # Postprocess tokens for visualization
    tokens = xai_output.input_tokens
    if args.model_family == "llama2":
        # move spaces to the previous token
        for i in range(1, len(tokens)):
            if tokens[i].startswith(" "):
                tokens[i - 1] += " "
                tokens[i] = tokens[i].lstrip()
    elif args.model_family == "llama3" or args.model_family == "unllama3":
        if (
            xai_output.xai_method == ExplainerMethod.SHAP_PARTITION
            or xai_output.xai_method == ExplainerMethod.SHAP_PARTITION_TFIDF
        ):
            for i in range(1, len(tokens)):
                if tokens[i].startswith(" "):
                    tokens[i - 1] += " "
                    tokens[i] = tokens[i].lstrip()
        else:
            cleaned = []
            for token in tokens:
                cleaned.append(
                    token.replace("âĢĻ", "'")
                    .replace("âĢľ", '"')
                    .replace("âĢĿ", "”")
                    .replace("Ġ", "")
                )
            tokens = cleaned

    # Prepare tokens for visualization (add underscores etc)
    tokens = clean_tokens(prepare_fixed_bert_tokens_for_pdf_viz(tokens))

    emphasize_deviations = (
        xai_output.xai_method == ExplainerMethod.SHAP_PARTITION
        or xai_output.xai_method == ExplainerMethod.SHAP_PARTITION_TFIDF,
    )
    # Create visualization and save it
    from app.utils.sdg import get_sdg_colormap

    # check if args.output_path is a directory
    output_path = args.output_path
    if os.path.isdir(output_path):
        output_path = os.path.join(
            output_path,
            f"{args.index}_{args.model_family}_{xai_output.xai_method.value}.pdf",
        )

    pdf_heatmap(
        tokens,
        token_scores[class_index],
        cmap=get_sdg_colormap(class_index + 1),
        emphasize_deviations=emphasize_deviations,
        path=output_path,
        backend="xelatex",
    )

    log.info("Visualization saved", predicted_label=xai_output.predicted_label)
    # Free up memory after visualization
    del token_scores
    torch.cuda.empty_cache()

    if args.evaluate:
        # Faithfulness evaluation
        xai_evaluator = XAIEvaluator(model, tokenizer, device)
        evaluation = xai_evaluator.evaluate_explanation(xai_output, class_index)

        # Create a dictionary with rounded scores
        faithfulness_scores = {
            e.name: round(e.score, 4) for e in evaluation.evaluation_scores
        }

        # Log the faithfulness scores as a single dictionary
        log.info(
            "Completed evaluating for faithfulness of explanation.",
            faithfulness_scores=faithfulness_scores,
            predicted_label=xai_output.predicted_label,
            model_family=args.model_family,
            method=xai_output.xai_method.value,
            sample_index=args.index,
        )


def evaluate(args, model, tokenizer, data_manager):
    # Model + explainer setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log = structlog.get_logger().bind(device=device)
    model = model.to(device)
    explainer = get_explainer(args.method, model, tokenizer, device, args)

    xai_evaluator = XAIEvaluator(model, tokenizer, device)

    # Load data into a DataFrame
    df = data_manager.load_data()

    evaluations_for_explanations, start_index = None, args.start_index or 0

    # check if starting index exists
    if args.start_index is not None:
        # validate start index
        assert args.start_index >= 0, "Start index must be greater than or equal to 0"
        assert args.start_index < len(
            df
        ), "Start index must be less than the number of samples"

        # load previous data
        evaluations_for_explanations = xai_evaluator.load_evaluation(
            model_name=args.model_family,
            xai_method=args.method,
            base_dir=args.output_path,
        )
    else:
        evaluations_for_explanations = np.zeros((len(df), model.num_labels, 3))

    explanations = []

    log.info("Starting faithfulness evaluation", n_samples=len(df))

    for sample_i in tqdm(range(start_index, len(df)), desc="Evaluating explanations"):
        abstract = df.iloc[sample_i].abstract
        abstract = truncate_to_bert_limit(abstract, tokenizer)
        xai_output = explainer.explain(abstract)
        explanations.append(xai_output)
        evaluations_for_explanations[sample_i] = xai_evaluator.evaluate_sample(
            xai_output
        )
        if args.output_path:
            xai_evaluator.save_evaluation(
                evaluations_for_explanations,
                model_name=args.model_family,
                xai_method=args.method,
                base_dir=args.output_path,
            )

    log.info(
        "Finished faithfulness evaluation",
        method=args.method,
        evaluation_shape=evaluations_for_explanations.shape,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="SDG text-classification explainer CLI",
        description="CLI for XAI explanation generation for SDG text classification models.",
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
    parser.add_argument(
        "--batch-size",
        help="Batch size for loading data.",
        type=int,
        default=500,
    )
    parser.add_argument(
        "-i",
        "--input-path",
        help="Path to the input data (abstracts, csv file).",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        help="Path to save the explanations, evaluation or visualization results.",
    )

    # Add subparsers for different actions ('explain', 'visualize', 'evaluate')
    subparsers = parser.add_subparsers(dest="action", required=True)

    # 'Explain' action / mode
    explain_parser = subparsers.add_parser(
        "explain", help="Explain predictions on given abstracts."
    )
    explain_parser.add_argument(
        "--n-workers",
        help="Number of workers PER GPU to use for parallel processing (computation of XAI values).",
        type=int,
        default=1,
    )
    explain_parser.add_argument(
        "--n-publications",
        help="Limit of publications to process in total.",
        type=int,
        default=100,
    )

    # 'Visualize' action / mode
    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize explanations into a pdf file."
    )
    visualize_parser.add_argument(
        "--index",
        help="Index of the publication to visualize. For local, use 0-based index, for Qdrant use point ID.",
        type=int,
        required=True,
    )
    visualize_parser.add_argument(
        "--class-index",
        help="Index of the class to visualize. (0-indexed SDG classes). If `None` then predicted class is used.",
        type=int,
        default=None,
    )
    visualize_parser.add_argument(
        "--evaluate",
        help="If set, evaluate faithfulness of the explanation after visualization.",
        action="store_true",
    )

    # 'Evaluate' action / mode
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate explanations")
    evaluate_parser.add_argument(
        "--start-index",
        help="Index of the first publication to evaluate (continue from), 0-indexed.",
        default=None,
        type=int,
    )

    args = parser.parse_args()

    # Validate args
    if args.method == "shap-partition-tfidf" and not args.tfidf_corpus_path:
        parser.error(
            "The --tfidf-corpus-path argument is required for the 'shap-partition-tfidf' method"
        )

    if args.data_source_target == DataSource.LOCAL and not args.input_path:
        parser.error(
            "The --input-path argument is required for the 'local' data source"
        )

    log.info("Parsed arguments, loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(args)
    log.info("Loaded model and tokenizer", model_family=args.model_family)

    # Setup data manager
    data_manager = None
    if args.data_source_target == DataSource.QDRANT:
        data_manager = QdrantManager(
            "publications", load_only_labelled_data=True, batch_size=args.batch_size
        )
    elif args.data_source_target == DataSource.LOCAL:
        data_manager = LocalManager(
            input_file_path=args.input_path,
            output_file_path=args.output_path,
            batch_size=args.batch_size,
        )
    log.info(
        "Data manager set up", data_source_and_target=args.data_source_target.value
    )

    log.info(
        f"Starting {args.action} action",
        method=args.method,
    )
    if args.action == "explain":
        explain(args, model, tokenizer, data_manager)
    elif args.action == "visualize":
        visualize(args, model, tokenizer, data_manager)
    elif args.action == "evaluate":
        evaluate(args, model, tokenizer, data_manager)
