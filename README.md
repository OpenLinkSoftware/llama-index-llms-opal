# Virtuoso OPAL LLamaIndex Integration

### Supported models:
-  "gpt-4o",
-  "gpt-4-vision-preview",
-  "gpt-4-turbo-preview",
-  "gpt-4-turbo",
-  "gpt-4-1106-preview",
-  "gpt-4-0613",
-  "gpt-4-0314",
-  "gpt-4-0125-preview",
-  "gpt-4",
-  "gpt-3.5-turbo-16k-0613",
-  "gpt-3.5-turbo-16k",
-  "gpt-3.5-turbo-1106",
-  "gpt-3.5-turbo-0613",
-  "gpt-3.5-turbo-0301",
-  "gpt-3.5-turbo-0125",
-  "gpt-3.5-turbo",
-  "ft:gpt-3.5-turbo-0125:openlink-software::939ZpZid",
-  "ft:gpt-3.5-turbo-0125:openlink-software::938VNgt1",
-  "dall-e-3",
-  "dall-e-2",

### Supported functions:
-  "DB.DBA.graphqlQuery",
-  "DB.DBA.graphqlEndpointQuery",
-  "UB.DBA.uda_howto",
-  "UB.DBA.sparqlQuery",
-  "DB.DBA.vos_howto_search",
-  "Demo.demo.execute_sql_query",
-  "Demo.demo.execute_spasql_query",

### Supported finetunes:
-  "system-virtuoso-support-assistant-config",
-  "system-virtuososupportassistantconfiglast",
-  "system-uda-support-assistant-config",
-  "system-udasupportassistantconfigtemp",
-  "system-www-support-assistant-config",
-  "system-data-twingler-config",

#### Default:
- **api_base**: "https://linkeddata.uriburner.com"
- **model**: "gpt-4o"
- **functions**: ["UB.DBA.sparqlQuery", "DB.DBA.vos_howto_search", "Demo.demo.execute_sql_query", "DB.DBA.graphqlQuery"]
-  **finetune**: "system-data-twingler-config"
-  **temperature**: 0.2
-  **top_p**: 0.5

#### OPAL class constructor parameters and default values:
```python
    def __init__(
        self,
        model: Optional[str] = "gpt-4o",
        finetune: Optional[str] = "system-data-twingler-config",
        funcs_list: Optional[list()] = ["UB.DBA.sparqlQuery", "DB.DBA.vos_howto_search", "Demo.demo.execute_sql_query", "DB.DBA.graphqlQuery"],
        api_base: Optional[str] = "https://linkeddata.uriburner.com",
        api_key: Optional[str] = None,
        openai_key: Optional[str] = None,

        temperature: Optional[float] = 0.2,
        top_p: Optional[float] = 0.5,
        max_tokens: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
```

#### Example:
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




#### How to get OPENLINK_API_KEY
Visit the API Key Generation Page at: https://linkeddata.uriburner.com/oauth/applications.vsp

### Authors
- 2024, OpenLink Software

