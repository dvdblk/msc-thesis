from lime.lime_text import LimeTextExplainer

from app.explainers.base import BaseExplainer


class LimeExplainer(BaseExplainer):

    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)

    def explain(self, text):
        explainer = LimeTextExplainer(class_names=self.model.config.id2label.values())
        explanation = explainer.explain_instance(
            text, self.predict_proba, num_features=6
        )
        return explanation
