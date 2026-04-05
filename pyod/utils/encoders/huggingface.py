# -*- coding: utf-8 -*-
"""HuggingFaceEncoder for EmbeddingOD."""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np

try:
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
except ImportError:
    torch = None
    AutoModel = None

from . import BaseEncoder


class HuggingFaceEncoder(BaseEncoder):
    """Encoder using HuggingFace transformers.

    Supports both text (AutoTokenizer + AutoModel) and image
    (AutoImageProcessor + AutoModel) modalities.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or path.

    device : str or None, optional (default=None)
        Device for inference. None for auto-detection.

    pooling : str, optional (default='cls')
        Pooling strategy: 'cls' for CLS token, 'mean' for
        mean of all token embeddings.

    modality : str, optional (default='text')
        Input modality: 'text' or 'image'.

    Examples
    --------
    >>> from pyod.utils.encoders.huggingface import HuggingFaceEncoder
    >>> encoder = HuggingFaceEncoder('bert-base-uncased', modality='text')
    >>> embeddings = encoder.encode(["hello", "world"])
    """

    def __init__(self, model_name, device=None, pooling='cls',
                 modality='text'):
        if AutoModel is None:
            raise ImportError(
                "HuggingFaceEncoder requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch")
        self.model_name = model_name
        self.device = device
        self.pooling = pooling
        self.modality = modality

    def _load_model(self):
        """Load model and processor/tokenizer on first use."""
        if self.device is None:
            if torch.cuda.is_available():
                self.device_ = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and \
                    torch.backends.mps.is_available():
                self.device_ = torch.device('mps')
            else:
                self.device_ = torch.device('cpu')
        else:
            self.device_ = torch.device(self.device)

        self.model_ = AutoModel.from_pretrained(
            self.model_name).to(self.device_)
        self.model_.eval()

        if self.modality == 'image':
            self.processor_ = AutoImageProcessor.from_pretrained(
                self.model_name)
        else:
            self.processor_ = AutoTokenizer.from_pretrained(
                self.model_name)

    def encode(self, X, batch_size=32, show_progress=True):
        """Encode text or images to embeddings.

        Parameters
        ----------
        X : list of str (text) or list of PIL.Image (image)
            Input data.

        batch_size : int, optional (default=32)
            Batch size for encoding.

        show_progress : bool, optional (default=True)
            Show progress bar.

        Returns
        -------
        embeddings : numpy array of shape (n_samples, n_features)
        """
        if not hasattr(self, 'model_'):
            self._load_model()

        all_embeddings = []

        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]

            if self.modality == 'image':
                inputs = self.processor_(
                    images=list(batch), return_tensors='pt'
                ).to(self.device_)
            else:
                inputs = self.processor_(
                    list(batch), return_tensors='pt',
                    padding=True, truncation=True, max_length=512
                ).to(self.device_)

            with torch.no_grad():
                outputs = self.model_(**inputs)

            hidden = outputs.last_hidden_state

            if self.pooling == 'cls':
                emb = hidden[:, 0, :]
            elif self.pooling == 'mean':
                if self.modality == 'text':
                    mask = inputs['attention_mask'].unsqueeze(-1).float()
                    emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                else:
                    emb = hidden[:, 1:, :].mean(dim=1)
            else:
                raise ValueError(
                    "pooling must be 'cls' or 'mean', got '%s'"
                    % self.pooling)

            all_embeddings.append(emb.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        return self._validate_output(embeddings, n_samples=len(X))
