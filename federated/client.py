from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import flwr.common as flwr_common
from flwr.client import NumPyClient
from flwr.server.client_proxy import ClientProxy
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from federated.parameter_utils import (
    ndarrays_to_flower_parameters,
    flower_parameters_to_ndarrays,
    ndarrays_to_lora_dict,
    lora_dict_to_ndarrays,
    compute_lora_delta,
)


import threading
MODEL_LOCK = threading.Lock()


class LoRADeltaClient(NumPyClient):
    """
    Flower NumPyClient that operates on LoRA parameter tensors and returns
    delta updates (ΔLoRA) per the architectural contract.

    Supports both:
    - Real local SFT step execution on TinyLlama + LoRA using PyTorch/PEFT (use_real_sft=True)
    - Deterministic synthetic parameter updates for fast protocol testing (use_real_sft=False)
    """

    def __init__(
        self,
        cid: str,
        num_examples: int,
        parameter_keys: Optional[List[str]] = None,
        parameter_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        synthetic_step_scale: float = 0.01,
        seed: int = 42,
        use_real_sft: bool = False,
        local_dataset: Optional[List[str]] = None,
        peft_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        learning_rate: float = 2e-4,
        max_seq_length: int = 128,
        device: str = "cpu",
    ):
        self.cid = cid
        self.num_examples = num_examples
        self.parameter_keys = parameter_keys or []
        self.parameter_shapes = parameter_shapes or {}
        self.synthetic_step_scale = synthetic_step_scale
        self.seed = seed
        self.use_real_sft = use_real_sft
        self.local_dataset = local_dataset or []
        self.peft_model = peft_model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.device = device
        self.current_parameters: Optional[List[np.ndarray]] = None
        self.fit_call_count = 0
        self.last_loss: Optional[float] = None
        self.last_delta_norm: Optional[float] = None

    def get_parameters(self, config: Dict[str, flwr_common.Scalar]) -> List[np.ndarray]:
        """
        Returns the client's current parameter state.
        """
        if self.current_parameters is not None:
            return [p.copy() for p in self.current_parameters]
        return []

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Stores the current global parameters sent by the server.
        """
        self.current_parameters = [p.copy() for p in parameters]

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, flwr_common.Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, flwr_common.Scalar]]:
        """
        Executes a local round. Computes ΔLoRA = local_updated - global.
        Returns the delta ndarrays, sample count, and training metrics.
        """
        self.set_parameters(parameters)
        self.fit_call_count += 1
        current_round = int(config.get("server_round", self.fit_call_count))

        if self.use_real_sft:
            return self._fit_real_sft(parameters, current_round)
        else:
            return self._fit_synthetic(parameters, current_round)

    def _fit_synthetic(
        self,
        parameters: List[np.ndarray],
        current_round: int,
    ) -> Tuple[List[np.ndarray], int, Dict[str, flwr_common.Scalar]]:
        """
        Generates deterministic synthetic delta matching the exact shapes of parameters.
        """
        cid_hash = abs(hash(self.cid)) % 10000
        client_seed = self.seed + cid_hash * 100 + current_round

        deltas: List[np.ndarray] = []
        for i, global_arr in enumerate(parameters):
            rng = np.random.default_rng(client_seed + i)
            delta = (rng.standard_normal(global_arr.shape) * self.synthetic_step_scale).astype(np.float32)
            deltas.append(delta)

        delta_norm = float(np.sqrt(sum(np.sum(d ** 2) for d in deltas)))
        self.last_delta_norm = delta_norm

        metrics: Dict[str, flwr_common.Scalar] = {
            "cid": self.cid,
            "round": current_round,
            "delta_norm": delta_norm,
            "num_examples": self.num_examples,
        }
        return deltas, self.num_examples, metrics

    def _fit_real_sft(
        self,
        parameters: List[np.ndarray],
        current_round: int,
    ) -> Tuple[List[np.ndarray], int, Dict[str, flwr_common.Scalar]]:
        """
        Executes exactly ONE real local SFT training step with TinyLlama + LoRA on CPU:
        1. Convert global ndarrays -> PyTorch LoRA dict
        2. Load global parameters into PEFT model via set_peft_model_state_dict
        3. Tokenize local training sample
        4. Run 1 forward + backward pass and optimizer step
        5. Extract updated local LoRA parameters via get_peft_model_state_dict
        6. Compute ΔLoRA = local - global
        7. Return ΔLoRA ndarrays, sample count, and training loss/norm
        """
        if self.peft_model is None or self.tokenizer is None:
            raise ValueError(f"Client {self.cid} requires peft_model and tokenizer for real SFT execution.")

        with MODEL_LOCK:
            # Ensure parameter keys are sorted and match the model state
            if not self.parameter_keys:
                current_lora = get_peft_model_state_dict(self.peft_model)
                self.parameter_keys = sorted(current_lora.keys())

            # 1. Reconstruct received global LoRA dictionary
            global_lora_dict = ndarrays_to_lora_dict(parameters, self.parameter_keys)

            # 2. Set global parameters into PEFT model
            set_peft_model_state_dict(self.peft_model, global_lora_dict)

            # 3. Select local training data for this step
            if self.local_dataset and len(self.local_dataset) > 0:
                step_idx = (self.fit_call_count - 1) % len(self.local_dataset)
                text_sample = self.local_dataset[step_idx]
            else:
                text_sample = "### Instruction:\nSolve 2 + 2.\n\n### Response:\n4."

            inputs = self.tokenizer(
                text_sample,
                return_tensors="pt",
                max_length=self.max_seq_length,
                truncation=True,
            )
            inputs["labels"] = inputs["input_ids"].clone()
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 4. Run exactly ONE optimizer step
            self.peft_model.train()
            optimizer = torch.optim.AdamW(
                [p for p in self.peft_model.parameters() if p.requires_grad],
                lr=self.learning_rate,
            )
            optimizer.zero_grad()
            outputs = self.peft_model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            self.last_loss = loss_val

            # 5. Extract updated local LoRA parameters
            local_lora = get_peft_model_state_dict(self.peft_model)
            local_lora = {k: v.detach().cpu().to(torch.float32) for k, v in local_lora.items()}

            # 6. Compute exact ΔLoRA = local_lora - global_lora
            delta_dict = compute_lora_delta(local_lora, global_lora_dict)
            deltas = [delta_dict[k].numpy() for k in self.parameter_keys]

            delta_norm = float(np.sqrt(sum(np.sum(d ** 2) for d in deltas)))
            self.last_delta_norm = delta_norm

            metrics: Dict[str, flwr_common.Scalar] = {
                "cid": self.cid,
                "round": current_round,
                "loss": loss_val,
                "delta_norm": delta_norm,
                "num_examples": self.num_examples,
            }
            return deltas, self.num_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, flwr_common.Scalar],
    ) -> Tuple[float, int, Dict[str, flwr_common.Scalar]]:
        """
        Minimal evaluation hook returning loss 0.0 and sample count.
        """
        return 0.0, self.num_examples, {"cid": self.cid}


class LocalClientProxy(ClientProxy):
    """
    ClientProxy wrapper enabling in-process execution of NumPyClient instances
    with Flower's Server and SimpleClientManager.
    """

    def __init__(self, cid: str, client: NumPyClient):
        super().__init__(cid)
        self.client = client

    def get_parameters(
        self, ins: flwr_common.GetParametersIns, timeout: Optional[float], group_id: Optional[int] = None
    ) -> flwr_common.GetParametersRes:
        ndarrays = self.client.get_parameters(ins.config)
        return flwr_common.GetParametersRes(
            status=flwr_common.Status(flwr_common.Code.OK, "Success"),
            parameters=ndarrays_to_flower_parameters(ndarrays),
        )

    def fit(
        self, ins: flwr_common.FitIns, timeout: Optional[float], group_id: Optional[int] = None
    ) -> flwr_common.FitRes:
        try:
            ndarrays = flower_parameters_to_ndarrays(ins.parameters)
            deltas, num_examples, metrics = self.client.fit(ndarrays, ins.config)
            return flwr_common.FitRes(
                status=flwr_common.Status(flwr_common.Code.OK, "Success"),
                parameters=ndarrays_to_flower_parameters(deltas),
                num_examples=num_examples,
                metrics=metrics,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return flwr_common.FitRes(
                status=flwr_common.Status(flwr_common.Code.FIT_NOT_IMPLEMENTED, str(e)),
                parameters=ndarrays_to_flower_parameters([]),
                num_examples=0,
                metrics={"error": str(e)},
            )

    def evaluate(
        self, ins: flwr_common.EvaluateIns, timeout: Optional[float], group_id: Optional[int] = None
    ) -> flwr_common.EvaluateRes:
        ndarrays = flower_parameters_to_ndarrays(ins.parameters)
        loss, num_examples, metrics = self.client.evaluate(ndarrays, ins.config)
        return flwr_common.EvaluateRes(
            status=flwr_common.Status(flwr_common.Code.OK, "Success"),
            loss=loss,
            num_examples=num_examples,
            metrics=metrics,
        )

    def reconnect(
        self, ins: flwr_common.ReconnectIns, timeout: Optional[float], group_id: Optional[int] = None
    ) -> flwr_common.DisconnectRes:
        return f     lwr_common.DisconnectRes(
            status=flwr_common.Status(flwr_common.Code.OK, "Success"),
            reason="Simulation completed",
        )

    def get_properties(
        self, ins: flwr_common.GetPropertiesIns, timeout: Optional[float], group_id: Optional[int] = None
    ) -> flwr_common.GetPropertiesRes:
        return flwr_common.GetPropertiesRes(
            status=flwr_common.Status(flwr_common.Code.OK, "Success"),
            properties={},
        )
