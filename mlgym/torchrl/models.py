import json
import requests
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase, TensorDictModule
from torch import nn
from typing import Any
from typing import Literal

# import models
from mlgym.agent.base import AgentArguments, BaseAgent
from mlgym.agent.models import MetagenModel as MLGymMetagenModel, ContextWindowExceededError, RateLimitExceededError, \
    APIError, ModelArguments, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type, \
    CostLimitExceededError, ContextWindowExceededError, APIError, retry, _MAX_RETRIES
from mlgym.utils.config import keys_config


## Models

def SubmitBaselineModel():
    # from mlgym/agent/models.py:SubmitBaselineModel
    return TensorDictModule(
        lambda: "DISCUSSION\nLet's reproduce the baseline method using the baseline script.\n\n```\npython baseline.py\n```\n",
        in_keys=[],
        out_keys=["action"]
    )


def SubmitBaselineRLModel():
    # from mlgym/agent/models.py:SubmitBaselineRLModel
    return TensorDictModule(
        lambda: "DISCUSSION\nLet's reproduce the baseline method using the baseline script.\n\n```\npython src/train.py\n```\n",
        in_keys=[],
        out_keys=["action"]
    )


def SubmitBaselineWrongModel():
    # from mlgym/agent/models.py:SubmitBaselineWrongModel
    return TensorDictModule(
        lambda: "DISCUSSION\nLet's reproduce the baseline method using the baseline script.\n\n```\npython baseline.py\n```\n\nNow let's look at the files in the curren directory.\n\n```\nls -a\n```\n",
        in_keys=[],
        out_keys=["action"]
    )


class MetagenModel(TensorDictModuleBase):
    MODELS = MLGymMetagenModel.MODELS
    SHORTCUTS = MLGymMetagenModel.SHORTCUTS
    in_keys = ["message"]  # message is produced by HistoryProcessor
    out_keys = ["action"]

    def __init__(self, args: ModelArguments, access_token: str | None = None, url_base: str | None = None):
        super().__init__()
        self.args = args
        self.access_token = access_token
        self.url_base = url_base
        self.api_model = self.SHORTCUTS[
            self.args.model_name] if self.args.model_name in self.SHORTCUTS else self.args.model_name
        self._setup_client()
        self.model_metadata = self.MODELS[self.api_model]

    def _setup_client(self):
        if self.url_base is None:
            self.url_base = "https://graph-genai.facebook.com/v20.0"
        self.endpoint = "chat_completions"
        if self.access_token is None:
            self.access_token = keys_config.get("METAGEN_ACCESS_TOKEN")

    @retry(
        wait=wait_random_exponential(min=60, max=180),
        reraise=True,
        stop=stop_after_attempt(_MAX_RETRIES),
        retry=retry_if_not_exception_type((CostLimitExceededError, ContextWindowExceededError, APIError)),
    )
    def forward(self, tensordict: TensorDictBase):
        messages = tensordict["message"]
        if self.api_model == "gpt-o1-evals":
            request_params: dict[str, Any] = {
                "access_token": self.access_token,
                "model": self.api_model,
                "messages": messages,
                "options": {
                    "decode_output": True,
                    "max_tokens": 4096
                }
            }
        else:
            request_params: dict[str, Any] = {
                "access_token": self.access_token,
                "model": self.api_model,
                "messages": messages,
                "options": {  # For available options see https://fburl.com/code/6u69j4hm
                    "temperature": self.args.temperature,  # default is 0.6
                    "top_p": self.args.top_p,  # default is 0.9
                    "decode_output": True,  # default is True,
                    # "custom_stop": {"stop_words": [{"text": "ch", "token": 331}]},  # default is None
                },
            }
        response = requests.post(f"{self.url_base}/{self.endpoint}", json=request_params)  # type: ignore
        response = json.loads(response.text)
        try:
            text = response["text"]
        except KeyError:
            if response.get("error", None) is not None:
                if response["error"]["code"] == "context_length_exceeded":
                    msg = f"Context window ({self.model_metadata['max_context']} tokens) exceeded"
                    raise ContextWindowExceededError(msg)
                elif response["error"]["code"] == 1:
                    msg = f"Intermittent error, like rate limit, retrying...: {response['error']}"
                    raise RateLimitExceededError(msg)
                msg = f"Error querying metagen model: {response['error']}"
                raise APIError(msg)
        finish_reason = response["finish_reason"]

        # TODO: log tokens in TensorDict
        input_tokens = response["usage"]["num_prompt_tokens"]
        output_tokens = response["usage"]["num_completion_tokens"]
        if finish_reason == "prompt_too_long":
            msg = f"Generation exceeded context window: {input_tokens + output_tokens} / {self.model_metadata['max_context']}"
            raise ContextWindowExceededError(msg)

        return text
