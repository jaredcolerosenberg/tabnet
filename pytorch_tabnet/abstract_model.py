from dataclasses import dataclass, field
from typing import List, Any, Dict
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from scipy.sparse import csc_matrix
from abc import abstractmethod
from pytorch_tabnet import tab_network
from pytorch_tabnet.utils import (
    PredictDataset,
    create_explain_matrix,
    validate_eval_set,
    filter_weights,
    create_dataloaders,
)
from pytorch_tabnet.callbacks import (
    CallbackContainer,
    History,
    EarlyStopping,
)
from pytorch_tabnet.metrics import MetricContainer
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader
import io
import json
from pathlib import Path
import shutil
import zipfile


@dataclass
class TabModel(BaseEstimator):
    """ Class for TabNet model

    Parameters
    ----------
        device_name: str
            'cuda' if running on GPU, 'cpu' if not, 'auto' to autodetect
    """

    n_d: int = 8
    n_a: int = 8
    n_steps: int = 3
    gamma: float = 1.3
    cat_idxs: List[int] = field(default_factory=list)
    cat_dims: List[int] = field(default_factory=list)
    cat_emb_dim: int = 1
    n_independent: int = 2
    n_shared: int = 2
    epsilon: float = 1e-15
    momentum: float = 0.02
    lambda_sparse: float = 1e-3
    seed: int = 0
    clip_value: int = 1
    verbose: int = 1
    optimizer_fn: Any = torch.optim.Adam
    optimizer_params: Dict = field(default_factory=dict(lr=2e-2))
    scheduler_fn: Any = None
    scheduler_params: Dict = field(default_factory=dict)
    mask_type: str = "sparsemax"
    input_dim: int = None
    output_dim: int = None
    device_name: str = "auto"

    def __post_init__(self):
        self.virtual_batch_size = 1024
        torch.manual_seed(self.seed)
        # Defining device
        if self.device_name == "auto":
            if torch.cuda.is_available():
                device_name = "cuda"
            else:
                device_name = "cpu"
        self.device = torch.device(device_name)
        print(f"Device used : {self.device}")

    def _set_network(self):
        self.network = tab_network.TabNet(
            self.input_dim,
            self.output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            device_name=self.device_name,
            mask_type=self.mask_type,
        ).to(self.device)

        self.reducing_matrix = create_explain_matrix(
            self.network.input_dim,
            self.network.cat_emb_dim,
            self.network.cat_idxs,
            self.network.post_embed_dim,
        )

    def _set_metrics(self, metrics, eval_names):
        """Set attributes relative to the metrics.

        Parameters
        ----------
        metrics : list of str
            List of eval metric names.
        eval_names : list of str
            List of eval set names.

        """
        metrics = metrics or [self.get_default_metric()]

        # Set metric container for each sets
        self._metric_container_dict = {
            "train": MetricContainer(metrics, prefix="train_")
        }
        for name in eval_names:
            self._metric_container_dict.update(
                {name: MetricContainer(metrics, prefix=f"{name}_")}
            )

        self._metrics = []
        self._metrics_names = []
        for _, metric_container in self._metric_container_dict.items():
            self._metrics.extend(metric_container.metrics)
            self._metrics_names.extend(metric_container.names)

        # Early stopping metric is the last eval metric
        self.early_stopping_metric = self._metrics_names[-1]

    def _set_callbacks(self, custom_callbacks):
        """Setup the callbacks functions.

        Parameters
        ----------
        callbacks : list of func
            List of callback functions.

        """
        # Setup default callbacks history and early stopping
        self.history = History(
            self, verbose=self._verbose
        )
        early_stopping = EarlyStopping(
            early_stopping_metric=self.early_stopping_metric,
            is_maximize=self._metrics[-1]._maximize,
            patience=self.patience,
        )
        callbacks = [self.history, early_stopping]
        if custom_callbacks:
            callbacks.append(custom_callbacks)
        self._callback_container = CallbackContainer(callbacks)
        self._callback_container.set_trainer(self)

    def _set_optimizer(self):
        """Setup optimizer."""
        self._optimizer = self.optimizer_fn(
            self.network.parameters(), lr=self.lr, **self.optimizer_params
        )

    def _set_scheduler(self):
        """Setup scheduler."""
        self._scheduler = None
        if self.scheduler_fn:
            self._scheduler = self.scheduler_fn(
                self._optimizer, **self.scheduler_params
            )

    def construct_loaders(
        self, X_train, y_train, eval_set
    ):
        """
        Returns
        -------
        train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
            Training and validation dataloaders
        -------
        """
        # all weights are not allowed for this type of model
        filter_weights(self.weights)
        y_train_mapped = self.prepare_target(y_train)
        for i, (X, y) in enumerate(eval_set):
            y_mapped = self.prepare_target(y, self.target_mapper)
            eval_set[i] = (X, y_mapped)

        train_dataloader, valid_dataloaders = create_dataloaders(
            X_train,
            y_train_mapped,
            eval_set,
            self.weights,
            self.batch_size,
            self.num_workers,
            self.drop_last,
        )
        return train_dataloader, valid_dataloaders

    def fit(
        self,
        X_train,
        y_train,
        eval_set=None,
        eval_name=None,
        eval_metric=None,
        loss_fn=None,
        weights=0,
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        callbacks=None,
    ):
        """Train a neural network stored in self.network
        Using train_dataloader for training data and
        valid_dataloader for validation.

        Parameters
        ----------
            X_train: np.ndarray
                Train set
            y_train : np.array
                Train targets
            eval_set: list of tuple
                List of eval tuple set (X, y).
                The last one is used for early stopping
            eval_name: list of str
                List of eval set names.
            eval_metric : list of str
                List of evaluation metrics.
                The last metric is used for early stopping.
            weights : bool or dictionnary
                0 for no balancing
                1 for automated balancing
                dict for custom weights per class
            max_epochs : int
                Maximum number of epochs during training
            patience : int
                Number of consecutive non improving epoch before early stopping
            batch_size : int
                Training batch size
            virtual_batch_size : int
                Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
            num_workers : int
                Number of workers used in torch.utils.data.DataLoader
            drop_last : bool
                Whether to drop last batch during training
            callbacks : list of callback function
                List of custom callbacks
        """
        # update model name

        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.update_fit_params(
            X_train,
            y_train,
            eval_set,
            loss_fn,
            weights,
        )

        # Validate and reformat eval set depending on training data
        eval_names, eval_set = validate_eval_set(eval_set, eval_name, X_train, y_train)

        train_dataloader, valid_dataloaders = self.construct_loaders(
            X_train,
            y_train,
            eval_set
        )

        self._set_network()
        self._set_metrics(eval_metric, eval_name)
        self._set_callbacks(callbacks)
        self._set_optimizer()
        self._set_scheduler()

        # Call method on_train_begin for all callbacks
        self._callback_container.on_train_begin()

        # Training loop over epochs
        for epoch_idx in range(self.max_epochs):

            # Call method on_epoch_begin for all callbacks
            self._callback_container.on_epoch_begin(epoch_idx)

            self._train_epoch(train_dataloader)

            # Apply predict epoch to all eval sets
            for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
                self._predict_epoch(eval_name, valid_dataloader)

            # Call method on_epoch_end for all callbacks
            self._callback_container.on_epoch_end(epoch_idx, self.history.batch_metrics)

            if self._stop_training:
                break

        # Call method on_train_end for all callbacks
        self._callback_container.on_train_end()
        self.network.eval()

        # compute feature importance once the best model is defined
        self._compute_feature_importances(train_dataloader)

    def save_model(self, path):
        """
        Saving model with two distinct files.
        """
        saved_params = {}
        for key, val in self.get_params().items():
            if isinstance(val, type):
                # Don't save torch specific params
                continue
            else:
                saved_params[key] = val

        # Create folder
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save models params
        with open(Path(path).joinpath("model_params.json"), "w", encoding="utf8") as f:
            json.dump(saved_params, f)

        # Save state_dict
        torch.save(self.network.state_dict(), Path(path).joinpath("network.pt"))
        shutil.make_archive(path, "zip", path)
        shutil.rmtree(path)
        print(f"Successfully saved model at {path}.zip")
        return f"{path}.zip"

    def load_model(self, filepath):

        try:
            with zipfile.ZipFile(filepath) as z:
                with z.open("model_params.json") as f:
                    loaded_params = json.load(f)
                with z.open("network.pt") as f:
                    try:
                        saved_state_dict = torch.load(f)
                    except io.UnsupportedOperation:
                        # In Python <3.7, the returned file object is not seekable (which at least
                        # some versions of PyTorch require) - so we'll try buffering it in to a
                        # BytesIO instead:
                        saved_state_dict = torch.load(io.BytesIO(f.read()))
        except KeyError:
            raise KeyError("Your zip file is missing at least one component")

        self.__init__(**loaded_params)

        self._set_network()
        self.network.load_state_dict(saved_state_dict)
        self.network.eval()
        return

    def _train_epoch(self, train_loader):
        """
        Trains one epoch of the network in self.network

        Parameters
        ----------
            train_loader: a :class: `torch.utils.data.Dataloader`
                DataLoader with train set
        """
        self.network.train()

        y_true = []
        y_score = []

        for batch_idx, (X, y) in enumerate(train_loader):

            self._callback_container.on_batch_begin(batch_idx)

            batch_outs, batch_logs = self._train_batch(X, y)

            self._callback_container.on_batch_end(batch_idx, batch_logs)

            y_true.append(y)
            y_score.append(batch_outs["scores"])

        y_true = np.vstack(y_true)
        y_score = np.vstack(y_score)

        epoch_logs = {'lr': self._optimizer.param_groups[-1]["lr"]}
        metrics_logs = self._metric_container_dict["train"](y_true, y_score)
        epoch_logs.update(metrics_logs)
        self.history.batch_metrics.update(epoch_logs)

        return

    def _train_batch(self, X, y):
        """
        Trains one batch of data

        Parameters
        ----------
            X: torch.tensor
                Owned products
            y: torch.tensor
                Targeted products
        """
        batch_logs = {"batch_size": X.shape[0]}

        X = X.to(self.device).float()
        y = y.to(self.device).float()

        self._optimizer.zero_grad()

        output, M_loss = self.network(X)

        loss = self.compute_loss(output, y)
        # Add the overall sparsity loss
        loss -= self.lambda_sparse * M_loss

        # Perform backward pass and optimization
        loss.backward()
        if self.clip_value:
            clip_grad_norm_(self.network.parameters(), self.clip_value)
        self._optimizer.step()

        batch_logs["loss"] = loss.cpu().detach().numpy().item()

        if self._scheduler is not None:
            self._scheduler.step()

        # output metrics
        batch_outs = {
            "y": y.cpu().detach().numpy(),
            "scores": output.cpu().detach().numpy(),
        }

        return batch_outs, batch_logs

    def _predict_epoch(self, name, loader):
        """
        Predict an epoch and update metrics.

        Parameters
        ----------
            name: str
                Name of the validation set
            loader: torch.utils.data.Dataloader
                    DataLoader with validation set
        """
        # Setting network on evaluation mode (no dropout etc...)
        self.network.eval()

        y_true = []
        y_score = []

        # Main loop
        for batch_idx, (X, y) in enumerate(loader):
            scores = self._predict_batch(X)
            y_true.append(y)
            y_score.append(scores)

        y_true = np.vstack(y_true)
        y_score = np.vstack(y_score)
        y_score = self.convert_score(y_score)

        metrics_logs = self._metric_container_dict[name](y_true, y_score)

        self.network.train()
        self.history.batch_metrics.update(metrics_logs)
        return

    def _predict_batch(self, X):
        """
        Predict one batch of data.

        Parameters
        ----------
            x: torch.tensor
                Owned products
            y: torch.tenso
                Targeted products
        Returns
        -------
            np.array
                model scores
        """
        X = X.to(self.device).float()

        # compute model output
        scores, _ = self.network(X)

        return scores.cpu().detach().numpy()

    def load_best_model(self):
        if self.best_network is not None:
            self.network = self.best_network

    @abstractmethod
    def compute_loss(self, y_score, y_true):
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
            data: a :tensor: `torch.Tensor`
                Input data
            target: a :tensor: `torch.Tensor`
                Target data

        Returns
        -------
            predictions: np.array
                Predictions of the regression problem or the last class
        """
        raise NotImplementedError("users must define predict to use this base class")

    @abstractmethod
    def get_default_metric(self):
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
            data: a :tensor: `torch.Tensor`
                Input data
            target: a :tensor: `torch.Tensor`
                Target data

        Returns
        -------
            predictions: np.array
                Predictions of the regression problem or the last class
        """
        raise NotImplementedError("users must define predict to use this base class")

    def predict(self, X):
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
            data: a :tensor: `torch.Tensor`
                Input data
            target: a :tensor: `torch.Tensor`
                Target data

        Returns
        -------
            predictions: np.array
                Predictions of the regression problem
        """
        self.network.eval()
        dataloader = DataLoader(PredictDataset(X),
                                batch_size=self.batch_size, shuffle=False)

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()
            output, M_loss = self.network(data)
            predictions = output.cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return self.predict_func(res)

    def explain(self, X):
        """
        Return local explanation

        Parameters
        ----------
            data: a :tensor: `torch.Tensor`
                Input data
            target: a :tensor: `torch.Tensor`
                Target data

        Returns
        -------
            M_explain: matrix
                Importance per sample, per columns.
            masks: matrix
                Sparse matrix showing attention masks used by network.
        """
        self.network.eval()

        dataloader = DataLoader(
            PredictDataset(X), batch_size=self.batch_size, shuffle=False
        )

        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()

            M_explain, masks = self.network.forward_masks(data)
            for key, value in masks.items():
                masks[key] = csc_matrix.dot(
                    value.cpu().detach().numpy(), self.reducing_matrix
                )

            if batch_nb == 0:
                res_explain = csc_matrix.dot(
                    M_explain.cpu().detach().numpy(), self.reducing_matrix
                )
                res_masks = masks
            else:
                res_explain = np.vstack(
                    [
                        res_explain,
                        csc_matrix.dot(
                            M_explain.cpu().detach().numpy(), self.reducing_matrix
                        ),
                    ]
                )
                for key, value in masks.items():
                    res_masks[key] = np.vstack([res_masks[key], value])
        return res_explain, res_masks

    def _compute_feature_importances(self, loader):
        self.network.eval()
        feature_importances_ = np.zeros((self.network.post_embed_dim))
        for data, targets in loader:
            data = data.to(self.device).float()
            M_explain, masks = self.network.forward_masks(data)
            feature_importances_ += M_explain.sum(dim=0).cpu().detach().numpy()

        feature_importances_ = csc_matrix.dot(
            feature_importances_, self.reducing_matrix
        )
        self.feature_importances_ = feature_importances_ / np.sum(feature_importances_)
