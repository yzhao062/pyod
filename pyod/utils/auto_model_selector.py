from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import importlib.resources
import json
import re


def load_model_analyses_labels_only():
    analyses = {}
    model_list = []

    resource_dir = importlib.resources.files("pyod.utils").joinpath("model_analysis_jsons")


    for file in resource_dir.iterdir():
        if file.suffix == ".json":
            model_name = file.stem 
            with file.open("r", encoding="utf-8") as f:
                try:
                    analysis = json.load(f)
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON for model {model_name}")
                    continue

            # 处理 strengths / weaknesses
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
    def __init__(self, dataset, additional_notes='not specified', api_key=None):
        """
        Initializes the AutoModelSelector with a dataset, additional notes, and an OpenAI API key.
        
        Parameters:
        - dataset (pd.DataFrame): The dataset to analyze.
        - additional_notes (str): Additional notes about the dataset.
        - api_key (str): OpenAI API key. If not provided, it will be loaded from the environment variable.
        """
        load_dotenv()  
        
        # Initialize class variables
        self.dataset = dataset
        self.additional_notes = additional_notes
        self.api_key = api_key or os.getenv("API_KEY")
        self.model_info, self.model_list = load_model_analyses_labels_only()
        self.selected_model = None
        self.reason = None
        self.gpt_response = None
    
    def call_gpt(self, prompt):
        """
        Calls the OpenAI GPT-4o API with the provided prompt and returns the response.
        
        Parameters:
        - prompt (str): The prompt to send to GPT-4o.
        
        Returns:
        - response (str): The assistant's reply from GPT-4o.
        """
        client = OpenAI(api_key=self.api_key)
        messages = [
            {"role": "system", "content": 'You are a well-trained data scientist specifically good at machine learning.'},
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
        Analyzes the dataset and generates a prompt for GPT-4 to get standardized tags describing the dataset.
        
        Returns:
        - prompt (str): The prompt generated based on the dataset analysis.
        """
        import pandas as pd
        import numpy as np
        from scipy.stats import skew, kurtosis

        data = pd.DataFrame(self.dataset)
        
        # Step 1: Capture dataset statistics
        stats = {
            "shape": data.shape,
            "data_type_counts": data.dtypes.value_counts().to_dict(),
            "overall_missing_values_ratio": data.isnull().mean().mean().round(4),
        }
        
        # For numeric columns, compute overall statistics
        numeric_cols = data.select_dtypes(include=np.number)
        if not numeric_cols.empty:
            # Drop rows with any missing values in numeric columns for accurate calculations
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
            "You are an expert in data analysis. Based on the following statistical summary of a dataset, "
            "provide a list of standardized tags that best describe this dataset's properties.\n\n"
            "Dataset Analysis:\n"
            f"- Shape (rows, columns): {stats['shape']}\n"
            f"- Data type counts: {stats['data_type_counts']}\n"
            f"- Overall missing values ratio: {stats['overall_missing_values_ratio']}\n"
            f"- Overall descriptive stats (numeric columns): {stats['overall_descriptive_stats']}\n"
        )
        
        prompt += (
            f"\nAdditional Information: {self.additional_notes}\n\n"
            "Using the following categories, define appropriate tags for this dataset:\n"
            "Data size: Choose from ['small', 'medium', 'large']\n"
            "Data type: Choose from ['images', 'text', 'tabular data', 'time series', 'audio', 'video']\n"
            "Domain: Choose from ['medical', 'finance', 'education', 'social media', 'retail', 'manufacturing', 'agriculture', 'technology', 'automotive', 'others']\n"
            "Characteristics: Choose from ['noisy data', 'high dimensionality', 'sparse data', 'imbalanced data', 'real-time data', 'low-signal data']\n"
            "Additional requirements: Choose from ['CPU', 'GPU', 'high memory', 'low memory']\n\n"
            "Return your response in JSON format with keys corresponding to each category and list of relevant tags."
        )
        
        return prompt

    
    def model_auto_select(self):
        """
        Selects the most suitable deep learning model based on the dataset tags and model analyses.

        Returns:
        - selected_model (str): The name of the selected model.
        - reason (str): The reason for selecting the model.
        """
        # Step 1: Analyze and tag dataset if not already done
        # get data info
        
        max_retries = 3
        
        if not hasattr(self, 'data_info'):
            prompt = self.analyze_and_tag_dataset()
            for attempt in range(max_retries):
                data_info_response = self.call_gpt(prompt)

                try:
                    # Extract the section within curly braces
                    match = re.search(r'\{(.*?)\}', data_info_response, re.DOTALL)
                    if match:
                        data_info_response = match.group(0)

                    print(data_info_response)
                    data_info = json.loads(data_info_response)
                    self.data_info = data_info  # Store data_info
                    
                    break  # Exit loop if successful
                except json.JSONDecodeError:
                    print(f"Attempt {attempt + 1}/{max_retries} failed to parse data_info.")
                    if attempt == max_retries - 1:
                        print("Max retries reached. Could not parse dataset information.")
                        return None, None
    
    
        
        # Step 2: Load model analyses and list if not already loaded
        if not hasattr(self, 'model_info') or not hasattr(self, 'model_list'):
            self.model_info, self.model_list = self.load_model_analyses_labels_only()

         # Step 3: Construct the model selection prompt
        if not hasattr(self, 'selection_prompt'):
            selection_prompt = f"""
        You are an expert in machine learning model selection.
    
        Based on the dataset properties and model analyses provided, recommend the most suitable model from the given model list. Only select a model from the model list.
    
        Dataset Tags:
        {json.dumps(self.data_info, indent=4)}
    
        Model Analyses:
        {json.dumps(self.model_info, indent=4)}
    
        Model List:
        {json.dumps(self.model_list, indent=4)}
    
        Please compare the dataset tags with the strengths and weaknesses of each model, and select the most suitable model from the model list. The best model should be the maximum alignment of strengths and minimum alignment of weekness. Provide your selection in JSON format with the following keys:
        - "selected_model": the name of the top-choice selected model (must be exactly one from the model list)
        - "reason": a brief explanation of why this model is the best choice, considering the dataset properties and model characteristics.
    
        Ensure that 'selected_model' is exactly one of the model names from the model list.
        """
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
                    print(f"Attempt {attempt + 1}/{max_retries} failed: Model '{selected_model}' not in the model list.")
            except json.JSONDecodeError:
                print(f"Attempt {attempt + 1}/{max_retries} failed to parse selection response.")
                if attempt == max_retries - 1:
                    print("Max retries reached. Could not complete model selection.")
                    return None, None

        print("The top model is: ", self.selected_model)
        print("Reason to choose this model: ", self.reason)

        return self.selected_model, self.reason
        
        
    
    
    def get_top_clf(self): 
        """
        Imports the library of the selected model and initializes the classifier.

        Returns:
        - clf: Initialized classifier object.
        """
        if self.selected_model is None:
            self.selected_model, _ = self.model_auto_select()
        
        # Initialize the classifier based on the selected model
        if self.selected_model == 'MO_GAAL':
            from pyod.models.mo_gaal import MO_GAAL
            clf = MO_GAAL(epoch_num=30, batch_size=32)
        elif self.selected_model == 'SO_GAAL':
            from pyod.models.so_gaal import SO_GAAL
            clf = SO_GAAL()
        elif self.selected_model == 'AutoEncoder':
            from pyod.models.auto_encoder import AutoEncoder
            clf = AutoEncoder()
        elif self.selected_model == 'VAE':
            from pyod.models.vae import VAE
            clf = VAE()
        elif self.selected_model == 'AnoGAN':
            from pyod.models.anogan import AnoGAN
            clf = AnoGAN()
        elif self.selected_model == 'DeepSVDD':
            from pyod.models.deep_svdd import DeepSVDD
            clf = DeepSVDD()
        elif self.selected_model == 'ALAD':
            from pyod.models.alad import ALAD
            clf = ALAD()
        elif self.selected_model == 'AE1SVM':
            from pyod.models.ae1svm import AE1SVM
            clf = AE1SVM()
        elif self.selected_model == 'DevNet':
            from pyod.models.devnet import DevNet
            clf = DevNet()
        elif self.selected_model == 'RGraph':
            from pyod.models.rgraph import RGraph
            clf = RGraph()
        elif self.selected_model == 'LUNAR':
            from pyod.models.lunar import LUNAR
            clf = LUNAR()
        else:
            # Default to AutoEncoder if model not recognized
            from pyod.models.auto_encoder import AutoEncoder
            clf = AutoEncoder()
        
        return clf


                
        