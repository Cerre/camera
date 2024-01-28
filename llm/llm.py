from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser


llm: Optional[LlamaCpp] = None
callback_manager: Any = None

model_file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # OR "llama-2-7b-chat.Q4_K_M.gguf"
template_tiny = """<|system|>
                   You are a smart mini computer named Raspberry Pi. 
                   Write a short but funny answer.</s>
                   <|user|>
                   {question}</s>
                   <|assistant|>"""
template_llama = """<s>[INST] <<SYS>>
                    You are a smart mini computer named Raspberry Pi.
                    Write a short but funny answer.</SYS>>
                    {question} [/INST]"""
template = template_tiny


def llm_init():
    """ Load large language model """
    global llm, callback_manager

    callback_manager = CallbackManager([StreamingCustomCallbackHandler()])
    llm = LlamaCpp(
        model_path=model_file,
        temperature=0.1,
        n_gpu_layers=0,
        n_batch=256,
        callback_manager=callback_manager,
        verbose=True,
    )


def llm_start(question: str):
    """ Ask LLM a question """
    global llm, template

    prompt = PromptTemplate(template=template, input_variables=["question"])
    chain = prompt | llm | StrOutputParser()
    chain.invoke({"question": question}, config={})