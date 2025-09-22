import os
from typing import Callable, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class BaseDialogAgent:
    def __init__(self, name: str, system_message: SystemMessage, model: ChatOpenAI):
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chat model to the current message history and returns the message.
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates the {message} spoken by {name} into message history.
        """
        self.message_history.append(f"{name}: {message}")


class DialogSimulator:
    def __init__(
        self,
        agents: list[BaseDialogAgent],
        selection_function=Callable[[int, List[BaseDialogAgent]], int],
    ):
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()
        self._step = 0

    def inject_message(self, name: str, message: str) -> None:
        """
        Injects a message into the conversation from a specific agent.
        """
        for agent in self.agents:
            agent.receive(name, message)

    def step(self) -> tuple[str, str]:
        """
        Selects the next speaker, gets their message, and broadcasts it to all agents.
        """
        speaker_index = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_index]
        message = speaker.send()
        for agent in self.agents:
            agent.receive(speaker.name, message)
        self._step += 1
        return speaker.name, message
