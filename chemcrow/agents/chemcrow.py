from typing import Optional

import langchain
from dotenv import load_dotenv
from langchain import PromptTemplate, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import ValidationError

# Import the GROQ chat model
from langchain_groq.chat_models import ChatGroq
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor

from .prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, REPHRASE_TEMPLATE, SUFFIX
from .tools import make_tools


def _make_llm(model, temp, streaming: bool = False):
    # set api key as env variable
    # Use ChatGroq for models starting with "groq"
    if model.startswith("groq"):
        llm = ChatGroq(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
    elif model.startswith("text-"):
        llm = langchain.OpenAI(
            temperature=temp,
            model_name=model,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,
        )
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm


class ChemCrow:
    def __init__(
        self,
        tools=None,
        model="llama-3.3-70b-versatile",  # Use a GROQ model identifier here
        tools_model="llama3-8b-8192",  # Similarly for tools_model if needed
        temp=0.1,
        max_iterations=40,
        verbose=True,
        streaming: bool = True,
        api_keys: dict = {},
        local_rxn: bool = False,
    ):
        """Initialize ChemCrow agent."""

        load_dotenv()
        try:
            self.llm = _make_llm(model, temp, streaming)
        except ValidationError:
            raise ValueError("Invalid API key or model configuration")

        if tools is None:
            tools_llm = _make_llm(tools_model, temp, openai_api_key, streaming)
            tools = make_tools(tools_llm, api_keys=api_keys, local_rxn=local_rxn, verbose=verbose)

        # Initialize agent
        self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=ChatZeroShotAgent.from_llm_and_tools(
                self.llm,
                tools,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
                question_prompt=QUESTION_PROMPT,
            ),
            verbose=True,
            max_iterations=max_iterations,
        )

        rephrase = PromptTemplate(
            input_variables=["question", "agent_ans"], template=REPHRASE_TEMPLATE
        )

        self.rephrase_chain = chains.LLMChain(prompt=rephrase, llm=self.llm)

    def run(self, prompt):
        outputs = self.agent_executor({"input": prompt})
        return outputs["output"]
