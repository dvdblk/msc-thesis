import json
import os
import pandas as pd
from sdgrs_xai.explainers.model import XAIOutput

class LocalManager:
    """For loading csv files from local disk and storing csv results in a pandas DataFrame."""

    def __init__(self, input_file_path: str, output_file_path: str, batch_size: int):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.batch_size = batch_size

        # buffer to store intermediate results of XAI outputs
        self._results_buffer = []
        self._output_header_written = os.path.exists(self.output_file_path) and os.path.getsize(self.output_file_path) > 0

    def load_input_data(self) -> pd.DataFrame:
        """Load input data from a CSV file."""
        return pd.read_csv(self.input_file_path)

    def add_result(self, publication: pd.Series, xai_output: XAIOutput):
        """Add the result of the XAI output to the DataFrame."""
        # create a dict with all the data that we want to store later
        result = {
            # Publication data
            "publication_id": publication.get("id", ""),
            "title": publication.get("title", ""),
            "text": xai_output.text,

            # XAI data
            "predicted_id": xai_output.predicted_id,
            "predicted_label": xai_output.predicted_label,
            "probabilities": json.dumps(xai_output.probabilities),  # Convert list to JSON string
            "token_scores": json.dumps(xai_output.token_scores),  # Convert 2D list to JSON string
            "xai_method": xai_output.xai_method.name,
            "created_at": xai_output.created_at.isoformat()
        }

        # Add to buffer
        self._results_buffer.append(result)

    def store_data(self):
        """Store accumulated buffer to CSV file."""
        if not self._results_buffer:
            return

        # Convert buffer to DataFrame
        df = pd.DataFrame(self._results_buffer)

        # Write to CSV
        mode = "a" if self._output_header_written else "w"
        df.to_csv(
            self.output_file_path,
            mode=mode,
            header=not self._output_header_written,
            index=False
        )

        # Update header status and clear buffer
        self._output_header_written = True
        self._results_buffer.clear()

    def close(self):
        # final flush of the buffer
        self.store_data()

class QdrantManager:
    pass