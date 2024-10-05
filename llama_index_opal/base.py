import os
import json
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import httpx
from httpx import Timeout
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)

from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    # stream_completion_response_to_chat_response,
)
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms.llm import LLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

DEFAULT_REQUEST_TIMEOUT = 60.0

OPAL_MODELS = (
    "gpt-4o",
    "gpt-4-vision-preview",
    "gpt-4-turbo-preview",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0613",
    "gpt-4-0314",
    "gpt-4-0125-preview",
    "gpt-4",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo",
    "ft:gpt-3.5-turbo-0125:openlink-software::939ZpZid",
    "ft:gpt-3.5-turbo-0125:openlink-software::938VNgt1",
    "dall-e-3",
    "dall-e-2",
)

OPAL_FUNCTIONS = (
    "DB.DBA.graphqlQuery",
    "DB.DBA.graphqlEndpointQuery",
    "UB.DBA.uda_howto",
    "UB.DBA.sparqlQuery",
    "DB.DBA.vos_howto_search",
    "Demo.demo.execute_sql_query",
    "Demo.demo.execute_spasql_query",
)

OPAL_FINETUNES = (
    "system-virtuoso-support-assistant-config",
    "system-virtuososupportassistantconfiglast",
    "system-uda-support-assistant-config",
    "system-udasupportassistantconfigtemp",
    "system-www-support-assistant-config",
    "system-data-twingler-config",
)

# def get_additional_kwargs(
#     response: Dict[str, Any], exclude: Tuple[str, ...]
# ) -> Dict[str, Any]:
#     return {k: v for k, v in response.items() if k not in exclude}

def get_OPAL_assistants_list(
        api_base: Optional[str] = "https://linkeddata.uriburner.com",
    ) -> []:
    openlink_api_key = os.environ["OPENLINK_API_KEY"]
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {openlink_api_key}",
    }

    _url = f"{api_base}/chat/api/assistants"
    with httpx.Client(timeout=Timeout(DEFAULT_REQUEST_TIMEOUT)) as client:
        response = client.get(
            url=_url,
            headers=headers
        )
        response.raise_for_status()
        return response.json()


class OPAL_Assistant(LLM):
    """
    OpenLink  LLM.

    Examples:
        ## `pip install llama-index-llms-opal`
        `pip install git+https://github.com/OpenLinkSoftware/llama-index-llms-opal.git`

        ```python
        import os
        os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxx"
        os.environ["OPENLINK_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

        from llama_index.llms.opal import OPAL_Assist

        llm = OPAL_Assistant()
        resp = llm.complete("Paul Graham is")
        print(resp)

        ```
    """

    assistant_id: str = Field(description="The assistant_id to use.",
        default="asst_IcfB5bT1ep4SQW5vbNFChnX4")
    model: str = Field(description="The Model to use.")
    temperature: float = Field(description="The temperature to use for sampling.",
        default=0.2,
        gte=0.0,
        lte=1.0,
    )
    top_p: float = Field(description="The top_p to use for sampling.",
        default=0.5,
        gte=0.0,
        lte=1.0,
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate.",
    )
    continue_thread: bool = Field(description="Continue thread with LLM calls",
        default=False)

    openai_key: str = Field(default=None, description="OpenAI API Key",)
    api_key: str = Field(default=None, description="OpenLink API Key",)
    api_base: str = Field(
        default="https://linkeddata.uriburner.com",
        description="The base URL for OPAL API.",
    )

    funcs_list = []

    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to llamafile API server",
    )

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional Kwargs for the OPAL model"
    )
    model_info: Dict[str, Any] = Field(
        default_factory=dict, description="Details about the selected model"
    )

    @property
    def _openlink_api_key(self) -> str:
        return self.api_key or os.environ["OPENLINK_API_KEY"]

    @property
    def _openai_api_key(self) -> str:
        return self.openai_key or os.environ["OPENAI_API_KEY"]

    headers: Dict[str, str] = Field(
        default_factory=dict, description="Headers for API requests."
    )

    _api_threads_url = PrivateAttr()
    _api_run_url = PrivateAttr()
    _thread_id = PrivateAttr()


    def __init__(
        self,
        assistant_id: Optional[str] = "asst_IcfB5bT1ep4SQW5vbNFChnX4",
        model: Optional[str] = "",
        funcs_list: Optional[list()] = [], # type: ignore
        api_base: Optional[str] = "https://linkeddata.uriburner.com",
        api_key: Optional[str] = None,
        openai_key: Optional[str] = None,

        temperature: Optional[float] = 0.2,
        top_p: Optional[float] = 0.5,
        continue_thread: bool = False,
        max_tokens: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        """Initialize params."""

        # if assistant_id is None:
        #     raise (ValueError("assistant_id name is empty"))

        additional_kwargs = additional_kwargs or {}

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            api_base=api_base,
            api_key=api_key,
            # model_info=self._model.get_details(),
            callback_manager=callback_manager,
            # base class
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self._thread_id = None
        self.continue_thread=continue_thread
        self.openai_key=openai_key
        self.api_key=api_key
        self.top_p = top_p
        self.assistant_id = assistant_id
        self.funcs_list = funcs_list
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self._openlink_api_key}",
        }
        messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        completion_to_prompt = completion_to_prompt or (lambda x: x)
        callback_manager = callback_manager or CallbackManager([])

        self._api_threads_url = f"{self.api_base}/chat/api/threads"
        self._api_run_url = f"{self.api_base}/chat/api/runAssistant"


    @classmethod
    def class_name(self) -> str:
        """Get Class Name."""
        return "OPAL_Assistant"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=16384, ## TODO ??
            num_output=self.max_tokens or -1,
            model_name=self.model,
            is_chat_model=False,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        payload = {
            "assistant_id": self.assistant_id,
            "apiKey": self._openai_api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "prompt": prompt
        }
        thread_id = self._thread_id if self.continue_thread else None

        if len(self.funcs_list)>0:
            payload["functions"] = self.funcs_list

        if self.model is not None and len(self.model)>0:
            payload["model"] = self.model

        if thread_id is None:
            _create_url = f"{self._api_threads_url}?apiKey={self._openai_api_key}"
            with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
                response = client.post(
                    url=_create_url,
                    json={},
                    headers=self.headers
                )
                response.raise_for_status()
                thread_id = response.text
                if thread_id is None:
                    raise (ValueError("Could not create Thread"))
                if self.continue_thread:
                    self._thread_id = thread_id

        payload["thread_id"] = thread_id

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            response = client.post(
                url=self._api_run_url,
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            raw = response.json()
            kind = raw.get("kind")
            message = raw.get("data", "")
            if kind is not None and kind == "error":
                raise (ValueError(message))
            return CompletionResponse(
                text=message,
                raw = raw
                )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        kwargs = kwargs if kwargs else {}
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError


    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        # kwargs = kwargs if kwargs else {}
        # return self.complete(prompt, **kwargs)
        payload = {
            "assistant_id": self.assistant_id,
            "apiKey": self._openai_api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "prompt": prompt
        }
        thread_id = self._thread_id if self.continue_thread else None

        if len(self.funcs_list) > 0:
            payload["functions"] = self.funcs_list

        if self.model is not None and len(self.model)>0:
            payload["model"] = self.model

        if thread_id is None:
            _create_url = f"{self._api_threads_url}?apiKey={self._openai_api_key}"
            async with httpx.AsyncClient(timeout=Timeout(self.request_timeout)) as client:
                response = await client.post(
                    url=_create_url,
                    json={},
                    headers=self.headers
                )
                response.raise_for_status()
                thread_id = response.text
                if thread_id is None:
                    raise (ValueError("Could not create Thread"))
                if self.continue_thread:
                    self._thread_id = thread_id

        payload["thread_id"] = thread_id

        async with httpx.AsyncClient(timeout=Timeout(self.request_timeout)) as client:
            response = await client.post(
                url=self._api_run_url,
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            raw = response.json()
            kind = raw.get("kind")
            message = raw.get("data", "")
            if kind is not None and kind == "error":
                raise (ValueError(message))
            return CompletionResponse(
                text=message,
                raw = raw
                )

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        kwargs = kwargs if kwargs else {}
        prompt = self.messages_to_prompt(messages)
        completion_response = self.acomplete(prompt, **kwargs)
        return completion_response_to_chat_response(completion_response)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError


class OPAL(LLM):
    """
    OpenLink  LLM.

    Examples:
        ## `pip install llama-index-llms-opal`
        `pip install git+https://github.com/OpenLinkSoftware/llama-index-llms-opal.git`

        ```python
        import os
        os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxx"
        os.environ["OPENLINK_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

        from llama_index.llms.opal import OPAL

        llm = OPAL()
        resp = llm.complete("Paul Graham is")
        print(resp)

        ```
    """

    model: str = Field(description="The Model to use.")
    temperature: float = Field(description="The temperature to use for sampling.",
        default=0.2,
        gte=0.0,
        lte=1.0,
    )
    top_p: float = Field(description="The top_p to use for sampling.",
        default=0.5,
        gte=0.0,
        lte=1.0,
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate.",
    )
    continue_chat: bool = Field(description="Continue chat with LLM calls",
        default=False)

    openai_key: str = Field(default=None, description="OpenAI API Key",)
    api_key: str = Field(default=None, description="OpenLink API Key",)
    api_base: str = Field(
        default="https://linkeddata.uriburner.com",
        description="The base URL for OPAL API.",
    )
    finetune: str = Field(description="Finetune mode", default="system-data-twingler-config")
    funcs_list = ["UB.DBA.sparqlQuery", "DB.DBA.vos_howto_search", "Demo.demo.execute_sql_query", "DB.DBA.graphqlQuery"]

    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to llamafile API server",
    )

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional Kwargs for the OPAL model"
    )
    model_info: Dict[str, Any] = Field(
        default_factory=dict, description="Details about the selected model"
    )

    @property
    def _openlink_api_key(self) -> str:
        return self.api_key or os.environ["OPENLINK_API_KEY"]

    @property
    def _openai_api_key(self) -> str:
        return self.openai_key or os.environ["OPENAI_API_KEY"]

    headers: Dict[str, str] = Field(
        default_factory=dict, description="Headers for API requests."
    )

    _api_url = PrivateAttr()
    _chat_id = PrivateAttr()

    def __init__(
        self,
        model: Optional[str] = "gpt-4o",
        finetune: Optional[str] = "system-data-twingler-config",
        funcs_list: Optional[list()] = ["UB.DBA.sparqlQuery", "DB.DBA.vos_howto_search", "Demo.demo.execute_sql_query", "DB.DBA.graphqlQuery"], # type: ignore
        api_base: Optional[str] = "https://linkeddata.uriburner.com",
        api_key: Optional[str] = None,
        openai_key: Optional[str] = None,

        temperature: Optional[float] = 0.2,
        top_p: Optional[float] = 0.5,
        continue_chat: bool = False,
        max_tokens: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        """Initialize params."""

        additional_kwargs = additional_kwargs or {}

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            api_base=api_base,
            api_key=api_key,
            # model_info=self._model.get_details(),
            callback_manager=callback_manager,
            # base class
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self._chat_id = None
        self.continue_chat=continue_chat
        self.openai_key=openai_key
        self.api_key=api_key
        self.top_p = top_p
        self.finetune = finetune
        self.funcs_list = funcs_list
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self._openlink_api_key}",
        }
        messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        completion_to_prompt = completion_to_prompt or (lambda x: x)
        callback_manager = callback_manager or CallbackManager([])

        self._api_url = f"{self.api_base}/chat/api/chatCompletion"


    @classmethod
    def class_name(self) -> str:
        """Get Class Name."""
        return "OPAL"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=16384, ## TODO ??
            num_output=self.max_tokens or -1,
            model_name=self.model,
            is_chat_model=False,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        payload = {
            "model": self.model,
            "type": "user",
            "apiKey": self._openai_api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        chat_id = self._chat_id if self.continue_chat else None

        payload["call"] = self.funcs_list
        payload["fine_tune"] = self.finetune

        if chat_id is None:
            payload["chat_id"] = self.finetune
            with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
                response = client.post(
                    url=self._api_url,
                    json=payload,
                    headers=self.headers,
                )
                response.raise_for_status()
                raw = response.json()
                chat_id = raw.get("chat_id")
                if chat_id is None:
                    raise (ValueError("Could not create Chat"))
                if self.continue_chat:
                    self._chat_id = chat_id

        payload["chat_id"] = chat_id
        payload["question"] = prompt

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            response = client.post(
                url=self._api_url,
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            raw = response.json()
            kind = raw.get("kind")
            message = raw.get("data", "")
            if kind is not None and kind == "error":
                raise (ValueError(message))
            return CompletionResponse(
                text=message,
                raw = raw
                )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        kwargs = kwargs if kwargs else {}
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError


    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        # kwargs = kwargs if kwargs else {}
        # return self.complete(prompt, **kwargs)
        payload = {
            "model": self.model,
            "type": "user",
            "apiKey": self._openai_api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        chat_id = self._chat_id if self.continue_chat else None

        payload["call"] = self.funcs_list
        payload["fine_tune"] = self.finetune

        if chat_id is None:
            payload["chat_id"] = self.finetune
            async with httpx.AsyncClient(timeout=Timeout(self.request_timeout)) as client:
                response = await client.post(
                    url=self._api_url,
                    json=payload,
                    headers=self.headers,
                )
                response.raise_for_status()
                raw = response.json()
                chat_id = raw.get("chat_id")
                if chat_id is None:
                    raise (ValueError("Could not create Chat"))
                if self.continue_chat:
                    self._chat_id = chat_id

        payload["chat_id"] = chat_id
        payload["question"] = prompt

        async with httpx.AsyncClient(timeout=Timeout(self.request_timeout)) as client:
            response = await client.post(
                url=self._api_url,
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            raw = response.json()
            kind = raw.get("kind")
            message = raw.get("data", "")
            if kind is not None and kind == "error":
                raise (ValueError(message))
            return CompletionResponse(
                text=message,
                raw = raw
                )

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        kwargs = kwargs if kwargs else {}
        prompt = self.messages_to_prompt(messages)
        completion_response = self.acomplete(prompt, **kwargs)
        return completion_response_to_chat_response(completion_response)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError
