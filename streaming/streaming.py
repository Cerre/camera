class StreamingCustomCallbackHandler(StreamingStdOutCallbackHandler):
    """ Callback handler for LLM streaming """

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """ Run when LLM starts running """
        print("<LLM Started>")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """ Run when LLM ends running """
        print("<LLM Ended>")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """ Run on new LLM token. Only available when streaming is enabled """
        print(f"{token}", end="")
        add_display_tokens(token)