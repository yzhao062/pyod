import importlib
import importlib.resources
import json
import os
import re
import warnings

warnings.warn(
    "AutoModelSelector is deprecated and will be removed in PyOD v2.3.0. "
    "Use pyod.utils.ad_engine.ADEngine instead.",
    FutureWarning,
    stacklevel=2
)


def _check_openai_dependency():
    """Check that openai is installed and exposes the OpenAI client."""
    try:
        from openai import OpenAI  # noqa: F401
    except (ImportError, AttributeError):
        raise ImportError(
            "AutoModelSelector requires 'openai' (with the OpenAI client). "
            "Install or upgrade with: pip install --upgrade openai"
        )


# Registry mapping model names to (module_path, class_name, kwargs).
# The keys must match the JSON file stems after normalization.
_MODEL_REGISTRY = {
    'MO_GAAL': ('pyod.models.mo_gaal', 'MO_GAAL',
                 {'epoch_num': 30, 'batch_size': 32}),
    'SO_GAAL': ('pyod.models.so_gaal', 'SO_GAAL', {}),
    'AutoEncoder': ('pyod.models.auto_encoder', 'AutoEncoder', {}),
    'VAE': ('pyod.models.vae', 'VAE', {}),
    'AnoGAN': ('pyod.models.anogan', 'AnoGAN', {}),
    'DeepSVDD': ('pyod.models.deep_svdd', 'DeepSVDD', {}),
    'ALAD': ('pyod.models.alad', 'ALAD', {}),
    'AE1SVM': ('pyod.models.ae1svm', 'AE1SVM', {}),
    'DevNet': ('pyod.models.devnet', 'DevNet', {}),
    'RGraph': ('pyod.models.rgraph', 'RGraph', {}),
    'LUNAR': ('pyod.models.lunar', 'LUNAR', {}),
}

# Map JSON file stems (which may contain hyphens/spaces) to the
# canonical registry keys used in _MODEL_REGISTRY.
_NAME_NORMALIZATION = {
    'MO-GAAL': 'MO_GAAL',
    'SO-GAAL': 'SO_GAAL',
    'Deep SVDD': 'DeepSVDD',
}


def _normalize_model_name(name):
    """Normalize a model name to its canonical registry key."""
    return _NAME_NORMALIZATION.get(name, name)


def load_model_analyses_labels_only():
    analyses = {}
    model_list = []

    resource_dir = importlib.resources.files("pyod.utils").joinpath(
        "model_analysis_jsons")

    for file in resource_dir.iterdir():
        if file.suffix == ".json":
            raw_name = file.stem
            model_name = _normalize_model_name(raw_name)
            with file.open("r", encoding="utf-8") as f:
                try:
                    analysis = json.load(f)
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON for model {model_name}")
                    continue

            # Extract strength/weakness labels
            strengths = analysis.get('strengths', [])
            weaknesses = analysis.get('weaknesses', [])
            strengths_labels = []
            for item in strengths:
                label = item.get('label') if isinstance(item, dict) else item
                strengths_labels.append(label)
            weaknesses_labels = []
            for item in weaknesses:
                label = item.get('label') if isinstance(item, dict) else item
                weaknesses_labels.append(label)

            analyses[model_name] = {
                'strengths': strengths_labels,
                'weaknesses': weaknesses_labels
            }
            model_list.append(model_name)

    return analyses, model_list


class AutoModelSelector:
    def __init__(self, dataset, additional_notes='not specified',
                 api_key=None):
        """
        Initializes the AutoModelSelector with a dataset, additional notes,
        and an OpenAI API key.

        Parameters
        ----------
        dataset : array-like
            The dataset to analyze.
        additional_notes : str, optional (default='not specified')
            Additional notes about the dataset.
        api_key : str, optional
            OpenAI API key. If not provided, it will be loaded from the
            OPENAI_API_KEY environment variable.
        """
        _check_openai_dependency()

        # Load .env only if no explicit API key was provided
        if api_key is None:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass

        # Initialize class variables
        self.dataset = dataset
        self.additional_notes = additional_notes
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv(
            "API_KEY")
        self.model_info, self.model_list = load_model_analyses_labels_only()
        self.selected_model = None
        self.reason = None
        self.gpt_response = None

    def call_gpt(self, prompt):
        """
        Calls the OpenAI GPT-4o API with the provided prompt and returns
        the response.

        Parameters
        ----------
        prompt : str
            The prompt to send to GPT-4o.

        Returns
        -------
        response : str
            The assistant's reply from GPT-4o.
        """
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        messages = [
            {"role": "system",
             "content": 'You are a well-trained data scientist specifically '
                        'good at machine learning.'},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        ).choices[0].message.content
        return response

    def analyze_and_tag_dataset(self):
        """
        Analyzes the dataset and generates a prompt for GPT-4o to get
        standardized tags describing the dataset.

        Returns
        -------
        prompt : str
            The prompt generated based on the dataset analysis.
        """
        import pandas as pd
        import numpy as np
        from scipy.stats import skew, kurtosis

        data = pd.DataFrame(self.dataset)

        # Step 1: Capture dataset statistics
        stats = {
            "shape": data.shape,
            "data_type_counts": data.dtypes.value_counts().to_dict(),
            "overall_missing_values_ratio": data.isnull().mean().mean().round(
                4),
        }

        # For numeric columns, compute overall statistics
        numeric_cols = data.select_dtypes(include=np.number)
        if not numeric_cols.empty:
            # Drop rows with any missing values for accurate calculations
            numeric_data = numeric_cols.dropna()

            overall_mean = numeric_data.values.flatten().mean().round(4)
            overall_std = numeric_data.values.flatten().std().round(4)
            overall_min = numeric_data.values.flatten().min().round(4)
            overall_max = numeric_data.values.flatten().max().round(4)
            overall_skewness = skew(numeric_data.values.flatten()).round(4)
            overall_kurtosis = kurtosis(numeric_data.values.flatten()).round(4)

            stats["overall_descriptive_stats"] = {
                "mean": overall_mean,
                "std_dev": overall_std,
                "min": overall_min,
                "max": overall_max,
                "skewness": overall_skewness,
                "kurtosis": overall_kurtosis
            }
        else:
            stats["overall_descriptive_stats"] = {}

        # Step 2: Build the prompt
        prompt = (
            "You are an expert in data analysis. Based on the following "
            "statistical summary of a dataset, provide a list of "
            "standardized tags that best describe this dataset's "
            "properties.\n\n"
            "Dataset Analysis:\n"
            f"- Shape (rows, columns): {stats['shape']}\n"
            f"- Data type counts: {stats['data_type_counts']}\n"
            f"- Overall missing values ratio: "
            f"{stats['overall_missing_values_ratio']}\n"
            f"- Overall descriptive stats (numeric columns): "
            f"{stats['overall_descriptive_stats']}\n"
        )

        prompt += (
            f"\nAdditional Information: {self.additional_notes}\n\n"
            "Using the following categories, define appropriate tags for "
            "this dataset:\n"
            "Data size: Choose from ['small', 'medium', 'large']\n"
            "Data type: Choose from ['images', 'text', 'tabular data', "
            "'time series', 'audio', 'video']\n"
            "Domain: Choose from ['medical', 'finance', 'education', "
            "'social media', 'retail', 'manufacturing', 'agriculture', "
            "'technology', 'automotive', 'others']\n"
            "Characteristics: Choose from ['noisy data', "
            "'high dimensionality', 'sparse data', 'imbalanced data', "
            "'real-time data', 'low-signal data']\n"
            "Additional requirements: Choose from ['CPU', 'GPU', "
            "'high memory', 'low memory']\n\n"
            "Return your response in JSON format with keys corresponding "
            "to each category and list of relevant tags."
        )

        return prompt

    def model_auto_select(self):
        """
        Selects the most suitable deep learning model based on the
        dataset tags and model analyses.

        Returns
        -------
        selected_model : str
            The name of the selected model.
        reason : str
            The reason for selecting the model.
        """
        max_retries = 3

        if not hasattr(self, 'data_info'):
            prompt = self.analyze_and_tag_dataset()
            for attempt in range(max_retries):
                data_info_response = self.call_gpt(prompt)

                try:
                    # Extract the section within curly braces
                    match = re.search(r'\{(.*?)\}', data_info_response,
                                      re.DOTALL)
                    if match:
                        data_info_response = match.group(0)

                    print(data_info_response)
                    data_info = json.loads(data_info_response)
                    self.data_info = data_info

                    break  # Exit loop if successful
                except json.JSONDecodeError:
                    print(
                        f"Attempt {attempt + 1}/{max_retries} failed to "
                        f"parse data_info.")
                    if attempt == max_retries - 1:
                        print(
                            "Max retries reached. Could not parse dataset "
                            "information.")
                        return None, None

        # Load model analyses and list if not already loaded
        if not hasattr(self, 'model_info') or not hasattr(self, 'model_list'):
            self.model_info, self.model_list = (
                load_model_analyses_labels_only())

        # Construct the model selection prompt
        if not hasattr(self, 'selection_prompt'):
            selection_prompt = (
                "You are an expert in machine learning model selection.\n\n"
                "Based on the dataset properties and model analyses "
                "provided, recommend the most suitable model from the "
                "given model list. Only select a model from the model "
                "list.\n\n"
                f"Dataset Tags:\n"
                f"{json.dumps(self.data_info, indent=4)}\n\n"
                f"Model Analyses:\n"
                f"{json.dumps(self.model_info, indent=4)}\n\n"
                f"Model List:\n"
                f"{json.dumps(self.model_list, indent=4)}\n\n"
                "Please compare the dataset tags with the strengths and "
                "weaknesses of each model, and select the most suitable "
                "model from the model list. The best model should have "
                "maximum alignment of strengths and minimum alignment of "
                "weaknesses. Provide your selection in JSON format with "
                "the following keys:\n"
                '- "selected_model": the name of the top-choice selected '
                "model (must be exactly one from the model list)\n"
                '- "reason": a brief explanation of why this model is the '
                "best choice, considering the dataset properties and model "
                "characteristics.\n\n"
                "Ensure that 'selected_model' is exactly one of the model "
                "names from the model list."
            )
            self.selection_prompt = selection_prompt

        for attempt in range(max_retries):
            selection_response = self.call_gpt(self.selection_prompt)
            try:
                # Extract the section within curly braces
                match = re.search(r'\{(.*?)\}', selection_response, re.DOTALL)
                if match:
                    selection_response = match.group(0)

                selection_result = json.loads(selection_response)
                selected_model = selection_result.get('selected_model')
                reason = selection_result.get('reason')

                # Validate the selected model
                if selected_model in self.model_list:
                    self.selected_model = selected_model
                    self.reason = reason

                    break
                else:
                    print(
                        f"Attempt {attempt + 1}/{max_retries} failed: "
                        f"Model '{selected_model}' not in the model list.")
            except json.JSONDecodeError:
                print(
                    f"Attempt {attempt + 1}/{max_retries} failed to parse "
                    f"selection response.")
                if attempt == max_retries - 1:
                    print(
                        "Max retries reached. Could not complete model "
                        "selection.")
                    return None, None

        print("The top model is: ", self.selected_model)
        print("Reason to choose this model: ", self.reason)

        return self.selected_model, self.reason

    def get_top_clf(self):
        """
        Imports the library of the selected model and initializes the
        classifier.

        Returns
        -------
        clf : object
            Initialized classifier object.
        """
        if self.selected_model is None:
            self.selected_model, _ = self.model_auto_select()

        if self.selected_model in _MODEL_REGISTRY:
            module_path, class_name, kwargs = (
                _MODEL_REGISTRY[self.selected_model])
            module = importlib.import_module(module_path)
            clf_class = getattr(module, class_name)
            clf = clf_class(**kwargs)
        else:
            # Default to AutoEncoder if model not recognized
            from pyod.models.auto_encoder import AutoEncoder
            clf = AutoEncoder()

        return clf
