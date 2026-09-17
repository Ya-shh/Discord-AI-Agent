import asyncio
import itertools
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from enum import Enum, auto
from typing import Any, Dict, Iterator, List, Optional, Tuple

import anthropic
import discord
from charset_normalizer import from_path
from discord import ButtonStyle, SelectOption, ui
from discord.ext import commands
from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.embedders.fastembed import (
    FastembedDocumentEmbedder,
    FastembedSparseDocumentEmbedder,
    FastembedSparseTextEmbedder,
    FastembedTextEmbedder,
)
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("qdrant-bot")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"Missing required environment variable: {name}")
    return value


USER_QUERY_CHANNEL_ID = int(require_env("USER_QUERY_CHANNEL_ID"))
MODERATOR_CHANNEL_ID = int(require_env("MODERATOR_CHANNEL_ID"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "disFY")
DOCS_DIR = os.getenv("DOCS_DIR", "QDR")

CLAUDE_MODEL = "claude-opus-5"
DENSE_MODEL = "snowflake/snowflake-arctic-embed-l"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
EMBEDDING_DIM = 1024
MAX_CHUNK_SIZE = 2048
MIN_CHUNK_SIZE = 100
HISTORY_LIMIT = 50
DISCORD_MESSAGE_LIMIT = 2000


LANGUAGE_PATTERNS = {
    "python": r"\bdef\s+\w+|\bclass\s+\w+|\bimport\s+\w+|\bfrom\s+\w+\s+import",
    "javascript": r"\bfunction\s+\w+|\bconst\s+\w+|\blet\s+\w+|\bvar\s+\w+|\bimport\s+.*\bfrom\b|\bexport\s+",
    "java": r"\bpublic\s+(class|interface)\s+\w+|\bimport\s+\w+(\.\w+)*;|\bpackage\s+\w+(\.\w+)*;|\b(public|private|protected)\s+\w+\s+\w+\(|\@Override",
    "csharp": r"\busing\s+\w+(\.\w+)*;|\bnamespace\s+\w+|\bpublic\s+class\s+\w+|\b(public|private|protected)\s+\w+\s+\w+\(",
    "rust": r"\bfn\s+\w+|\blet\s+mut\s+\w+|\buse\s+\w+::\w+|\bpub\s+(struct|enum|trait)\s+\w+|\bimpl\s+\w+",
    "http": r"^(get|post|put|delete|patch|head|options|trace)\s+/\S*|^\s*(\w+):\s*(\{|\[|\'|\"|\d)",
    "bash": r"\b(?:#!/bin/bash|#!/usr/bin/env bash)|\bcurl\s+-[X]\s+\w+|\b(?:if|for|while)\s+\[|(?:\b|\$)(?:echo|cd|pwd|ls|mkdir|rm|cp|mv)",
}

CHUNK_PATTERNS = {
    "python": r"(def\s+\w+|class\s+\w+|import\s+\w+|\w+\s*=|if\s+__name__\s*==\s*['\"]__main__['\"])",
    "javascript": r"(function\s+\w+|\w+\s*=\s*function|\w+\s*\(.*?\)\s*\{|class\s+\w+|import\s+.*?from|export\s+)",
    "java": r"(public\s+[\w<>\[\]]+\s+\w+\s*\(.*?\)|class\s+\w+|interface\s+\w+|enum\s+\w+|import\s+)",
    "csharp": r"(public\s+[\w<>\[\]]+\s+\w+\s*\(.*?\)|class\s+\w+|interface\s+\w+|enum\s+\w+|namespace\s+\w+|using\s+)",
    "rust": r"(fn\s+\w+|impl\s+[\w<>]+|struct\s+\w+|enum\s+\w+|trait\s+\w+|use\s+|mod\s+\w+)",
    "http": r"(^\s*(?:get|post|put|delete|patch|head|options|trace)\s+/\S*)",
    "bash": r"(#!/bin/bash|#!/usr/bin/env bash|\bcurl\s+-[X]\s+\w+|\b(?:if|for|while)\s+\[|(?:\b|\$)(?:function)\s+\w+)",
}


def detect_language(content: str) -> str:
    scores = {lang: len(re.findall(pattern, content, re.MULTILINE)) for lang, pattern in LANGUAGE_PATTERNS.items()}
    language, score = max(scores.items(), key=lambda item: item[1])
    return language if score > 0 else "text"


def normalize_whitespace(content: str) -> str:
    lines = (line.rstrip() for line in content.strip().splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def chunk_document(content: str, language: str) -> List[str]:
    pattern = CHUNK_PATTERNS.get(language)
    if pattern:
        # re.split keeps the captured separators; attach each one to the text that follows it
        parts = re.split(pattern, content, flags=re.MULTILINE)
        pieces = [parts[0]] + [parts[i] + parts[i + 1] for i in range(1, len(parts) - 1, 2)]
    else:
        pieces = [content]

    chunks, current = [], ""
    for piece in pieces:
        if len(current) >= MIN_CHUNK_SIZE and len(current) + len(piece) > MAX_CHUNK_SIZE:
            chunks.append(current)
            current = ""
        current += piece
        while len(current) > MAX_CHUNK_SIZE:
            chunks.append(current[:MAX_CHUNK_SIZE])
            current = current[MAX_CHUNK_SIZE:]
    chunks.append(current)

    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) >= MIN_CHUNK_SIZE]
    return chunks or [content.strip()]


def read_text_file(path: str) -> Optional[str]:
    best = from_path(path).best()
    if best is None:
        logger.warning("Could not detect encoding for %s, skipping.", path)
        return None
    return str(best)


def load_documents(directory: str) -> Iterator[Document]:
    counts: Counter = Counter()
    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            path = os.path.join(root, filename)
            content = read_text_file(path)
            if not content or not content.strip():
                continue
            language = detect_language(content)
            counts[language] += 1
            for index, chunk in enumerate(chunk_document(normalize_whitespace(content), language)):
                yield Document(content=chunk, meta={"source": path, "chunk_index": index, "language": language})
    logger.info("Loaded %d files: %s", sum(counts.values()), dict(counts))


document_store = QdrantDocumentStore(
    url=QDRANT_URL,
    api_key=Secret.from_env_var("QDRANT_API_KEY", strict=False),
    index=COLLECTION_NAME,
    embedding_dim=EMBEDDING_DIM,
    use_sparse_embeddings=True,
    hnsw_config={"m": 50, "ef_construct": 300},
    timeout=180,
)


def ingest_documents(directory: str, batch_size: int = 32) -> None:
    if not os.path.isdir(directory):
        logger.error("Docs directory '%s' not found, nothing to index.", directory)
        return

    pipeline = Pipeline()
    pipeline.add_component("sparse_embedder", FastembedSparseDocumentEmbedder(model=SPARSE_MODEL, progress_bar=False))
    pipeline.add_component("dense_embedder", FastembedDocumentEmbedder(model=DENSE_MODEL, progress_bar=False))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))
    pipeline.connect("sparse_embedder.documents", "dense_embedder.documents")
    pipeline.connect("dense_embedder.documents", "writer.documents")

    documents = load_documents(directory)
    written = 0
    while batch := list(itertools.islice(documents, batch_size)):
        written += pipeline.run({"sparse_embedder": {"documents": batch}})["writer"]["documents_written"]
        logger.info("Indexed %d chunks so far...", written)
    logger.info("Indexed %d chunks into '%s'.", written, COLLECTION_NAME)


def build_query_pipeline() -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("sparse_text_embedder", FastembedSparseTextEmbedder(model=SPARSE_MODEL, progress_bar=False))
    pipeline.add_component(
        "dense_text_embedder",
        FastembedTextEmbedder(
            model=DENSE_MODEL,
            prefix="Represent this sentence for searching relevant passages: ",
            progress_bar=False,
        ),
    )
    pipeline.add_component("retriever", QdrantHybridRetriever(document_store=document_store))
    pipeline.connect("sparse_text_embedder.sparse_embedding", "retriever.query_sparse_embedding")
    pipeline.connect("dense_text_embedder.embedding", "retriever.query_embedding")
    return pipeline


CODE_SYSTEM_PROMPT = """You are a technical assistant specializing in code implementation.
Guidelines:
1. Use exact code from the source documents
2. Maintain all variable names and structure
3. Include necessary context and setup
4. Provide clear explanations of the code
5. Focus on practical implementation details"""

CODE_USER_PROMPT = """Query: {query}

Available Code:
{code}

Context:
{context}

Technical Level: {level}
Response Requirements:
1. Show the complete implementation
2. Explain how the code works
3. Include any necessary setup
4. Maintain original variable names
5. Preserve code structure

Please provide a detailed response."""

CONCEPT_SYSTEM_PROMPT = """You are a technical documentation assistant.
Guidelines:
1. Provide clear, accurate explanations
2. Use terminology from the source documents
3. Include examples when relevant
4. Maintain technical accuracy
5. Focus on conceptual understanding"""

CONCEPT_USER_PROMPT = """Query: {query}

Source Content:
{content}

Key Concepts:
{concepts}

Previous Context:
{history}

Technical Level: {level}
Response Type: {query_type}

Please provide a focused explanation."""

CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
HOW_TO_RE = re.compile(
    r"how (?:do|can|to|would|should) (?:i|you|we)|what's the (?:best )?way to|show me (?:how|the way)"
    r"|example of|implement|create|build|write"
)
WHAT_IS_RE = re.compile(r"what (?:is|are|does)|explain|describe|tell me about|define|understand")
DEFINITION_RE = re.compile(r"\b(?:is|are|means|refers|represents)\b", re.IGNORECASE)
CODE_CONTEXT_TERMS = ("return", "parameter", "function", "method", "class")
BASIC_TERMS = {"simple", "basic", "beginner", "start", "example"}
ADVANCED_TERMS = {"advanced", "complex", "optimize", "performance", "efficient"}


def analyze_query(query: str, documents: List[Document]) -> Dict[str, Any]:
    query_lower = query.lower()
    terms = set(query_lower.split())
    code_text = " ".join(block for doc in documents for block in CODE_BLOCK_RE.findall(doc.content or "")).lower()

    if HOW_TO_RE.search(query_lower):
        query_type = "how-to"
    elif WHAT_IS_RE.search(query_lower):
        query_type = "what-is"
    else:
        query_type = "unknown"

    return {
        "query_type": query_type,
        "needs_code": query_type != "unknown" and bool(code_text) and any(term in code_text for term in terms),
        "level": "Advanced" if terms & ADVANCED_TERMS and not terms & BASIC_TERMS else "Basic",
    }


def format_history(history: List[Dict[str, Any]]) -> str:
    lines = []
    for exchange in history[-3:]:
        if exchange.get("user") and exchange.get("bot"):
            lines += [f"Human: {exchange['user']}", f"Assistant: {exchange['bot']}"]
    return "\n".join(lines) or "No prior conversation."


class ResponseGenerator:
    def __init__(self, query_pipeline: Pipeline):
        self.query_pipeline = query_pipeline
        self.client = anthropic.AsyncAnthropic(max_retries=5)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    async def retrieve(self, query: str) -> List[Document]:
        result = await asyncio.to_thread(
            self.query_pipeline.run,
            {"sparse_text_embedder": {"text": query}, "dense_text_embedder": {"text": query}},
        )
        return result["retriever"]["documents"]

    async def generate_response(self, query: str, history: List[Dict[str, Any]]) -> str:
        try:
            documents = await self.retrieve(query)
            if not documents:
                return "I don't have enough information to answer this question."

            analysis = analyze_query(query, documents)
            if analysis["needs_code"]:
                return await self._code_response(query, documents, analysis)
            return await self._conceptual_response(query, documents, analysis, history)
        except Exception:
            logger.exception("Error generating response")
            return "Error retrieving information."

    async def _code_response(self, query: str, documents: List[Document], analysis: Dict[str, Any]) -> str:
        code_blocks, context = [], []
        for doc in documents:
            code_blocks.extend(CODE_BLOCK_RE.findall(doc.content))
            prose = CODE_BLOCK_RE.sub("", doc.content)
            context.extend(
                sentence for sentence in SENTENCE_SPLIT_RE.split(prose)
                if any(term in sentence.lower() for term in CODE_CONTEXT_TERMS)
            )

        prompt = CODE_USER_PROMPT.format(
            query=query,
            code="\n\n".join(code_blocks),
            context=" ".join(context),
            level=analysis["level"],
        )
        return await self._complete(CODE_SYSTEM_PROMPT, prompt)

    async def _conceptual_response(
        self, query: str, documents: List[Document], analysis: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> str:
        content, concepts = [], []
        for doc in documents:
            prose = CODE_BLOCK_RE.sub("", doc.content)
            for sentence in SENTENCE_SPLIT_RE.split(prose):
                (concepts if DEFINITION_RE.search(sentence) else content).append(sentence)

        prompt = CONCEPT_USER_PROMPT.format(
            query=query,
            content=" ".join(content),
            concepts=" ".join(concepts),
            history=format_history(history),
            level=analysis["level"],
            query_type=analysis["query_type"],
        )
        return await self._complete(CONCEPT_SYSTEM_PROMPT, prompt)

    async def _complete(self, system: str, prompt: str) -> str:
        response = await self.client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            # Re-run the request on Anthropic's recommended fallback model if it gets declined
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            logger.warning("Claude declined the request.")
            return "Sorry, I can't help with that request."
        if response.stop_reason == "max_tokens":
            logger.warning("Response hit max_tokens and may be truncated.")
        return "".join(block.text for block in response.content if block.type == "text")


class ConversationalSystem:
    def __init__(self, generator: ResponseGenerator):
        self.generator = generator
        self.histories: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    async def handle_query(
        self, context_key: str, query: str, is_moderator_query: bool = False, update_history: bool = True
    ) -> str:
        history = self.histories[context_key]
        response = await self.generator.generate_response(query, history)
        if update_history:
            history.append({"user": query, "bot": response, "is_moderator_query": is_moderator_query})
            del history[:-HISTORY_LIMIT]
        return response


query_pipeline = build_query_pipeline()
conversational_system = ConversationalSystem(ResponseGenerator(query_pipeline))


class QueryBot(commands.Bot):
    async def setup_hook(self) -> None:
        await self.tree.sync()
        logger.info("Slash commands synced.")


intents = discord.Intents.default()
intents.message_content = True
bot = QueryBot(command_prefix="!", intents=intents)

thread_mapping: Dict[str, Dict[str, Any]] = defaultdict(dict)
original_author_mapping: Dict[str, str] = {}
muted_threads: set = set()

EMOJI_NUMBERS = ["0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]


def number_to_emoji(number: int) -> str:
    return EMOJI_NUMBERS[number] if 1 <= number <= 10 else str(number)


def find_context_key(thread_id: int) -> Optional[str]:
    for key, value in thread_mapping.items():
        threads = (value.get("moderator_thread"), value.get("user_thread"))
        if any(thread is not None and thread.id == thread_id for thread in threads):
            return key
    return None


def split_message(content: str, max_length: int = DISCORD_MESSAGE_LIMIT) -> List[str]:
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    pieces = []
    for paragraph in re.split(r"(\n\n+)", content):
        if len(paragraph) <= max_length:
            pieces.append(paragraph)
            continue
        for sentence in re.split(r"([.!?]\s)", paragraph):
            pieces.extend(sentence[i:i + max_length] for i in range(0, len(sentence), max_length))

    messages, current = [], ""
    for piece in pieces:
        if len(current) + len(piece) > max_length:
            messages.append(current)
            current = ""
        current += piece
    messages.append(current)
    return [message for message in messages if message.strip()]


async def safe_send_message(
    channel: discord.abc.Messageable, content: Optional[str], view: Optional[ui.View] = None
) -> Optional[discord.Message]:
    if not content or not content.strip():
        content = "I'm sorry, but I don't have any content to send at the moment."
    try:
        parts = split_message(content)
        message = None
        for i, part in enumerate(parts):
            message = await channel.send(part, view=view if i == len(parts) - 1 else None)
        return message
    except discord.Forbidden:
        logger.error("Bot does not have permission to send messages in this channel.")
    except discord.HTTPException:
        logger.exception("Failed to send message")
        try:
            return await channel.send("I encountered an error while trying to send a message. Please try again later.")
        except discord.HTTPException:
            pass
    return None


class SpinnerState(Enum):
    INITIALIZING = auto()
    PROCESSING = auto()
    SENDING = auto()


SPINNER_MESSAGES = {
    SpinnerState.INITIALIZING: "Preparing to process...",
    SpinnerState.PROCESSING: "Processing your request...",
    SpinnerState.SENDING: "Sending response...",
}


class ProcessStep:
    def __init__(self, description: str):
        self.description = description
        self.start_time = time.time()
        self.end_time: Optional[float] = None


class SpinnerView(ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @ui.button(label="Cancel", style=ButtonStyle.danger)
    async def cancel(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.defer()
        await spinner_manager.stop_spinner(interaction.channel)
        await interaction.followup.send("Process cancelled.", ephemeral=True)


class SpinnerManager:
    FRAMES = ("🔄", "🔁", "🔂")
    COLORS = (34, 35, 36)
    UPDATE_INTERVAL = 2

    def __init__(self):
        self.spinners: Dict[int, Dict[str, Any]] = {}

    async def start_spinner(self, channel: discord.abc.Messageable, state: SpinnerState) -> None:
        try:
            message = await channel.send(self._render(0, state, 0, []), view=SpinnerView())
        except discord.HTTPException as e:
            logger.error("Failed to start spinner in channel %s: %s", channel.id, e)
            return

        spinner = {"message": message, "state": state, "steps": [], "start_time": time.time()}
        previous = self.spinners.pop(channel.id, None)
        self.spinners[channel.id] = spinner
        spinner["task"] = asyncio.create_task(self._animate(channel.id, spinner))
        if previous:
            await self._dispose(previous)

    async def update_spinner_state(
        self, channel: discord.abc.Messageable, state: SpinnerState, step_description: Optional[str] = None
    ) -> None:
        spinner = self.spinners.get(channel.id)
        if spinner:
            spinner["state"] = state
            if step_description:
                spinner["steps"].append(ProcessStep(step_description))

    async def complete_step(self, channel: discord.abc.Messageable, step_description: str) -> None:
        spinner = self.spinners.get(channel.id)
        if not spinner:
            return
        step = next((s for s in spinner["steps"] if s.description == step_description and not s.end_time), None)
        if step:
            step.end_time = time.time()

    async def stop_spinner(self, channel: discord.abc.Messageable) -> None:
        spinner = self.spinners.pop(channel.id, None)
        if spinner:
            await self._dispose(spinner)

    async def _dispose(self, spinner: Dict[str, Any]) -> None:
        if spinner["task"] is not asyncio.current_task():
            spinner["task"].cancel()
        try:
            await spinner["message"].delete()
        except discord.NotFound:
            pass
        except discord.HTTPException as e:
            logger.warning("Failed to delete spinner message: %s", e)

    async def _animate(self, channel_id: int, spinner: Dict[str, Any]) -> None:
        index = 0
        while True:
            await asyncio.sleep(self.UPDATE_INTERVAL)
            if self.spinners.get(channel_id) is not spinner:
                return
            index += 1
            elapsed = int(time.time() - spinner["start_time"])
            try:
                await spinner["message"].edit(content=self._render(index, spinner["state"], elapsed, spinner["steps"]))
            except discord.NotFound:
                if self.spinners.get(channel_id) is spinner:
                    del self.spinners[channel_id]
                return
            except discord.HTTPException as e:
                logger.warning("Failed to update spinner in channel %s: %s", channel_id, e)

    def _render(self, index: int, state: SpinnerState, elapsed: int, steps: List[ProcessStep]) -> str:
        bold, reset = "[1m", "[0m"
        color = f"[{self.COLORS[index % len(self.COLORS)]}m"
        frame = self.FRAMES[index % len(self.FRAMES)]
        lines = [
            "```ansi",
            f"{bold}{color}{frame} {SPINNER_MESSAGES[state]}  {reset}",
            f"{bold}Time elapsed: {elapsed}s{reset}",
        ]
        if steps:
            lines += ["", "Progress:"]
            for step in steps:
                status = f"✅ Done in {step.end_time - step.start_time:.2f}s" if step.end_time else "🚀 In progress..."
                lines.append(f"- {step.description}: {status}")
        lines.append("```")
        return "\n".join(lines)


spinner_manager = SpinnerManager()


async def find_thread(channel: discord.ForumChannel, name: str) -> Optional[discord.Thread]:
    for thread in channel.threads:
        if thread.name == name:
            return thread
    async for thread in channel.archived_threads():
        if thread.name == name:
            return thread
    return None


async def find_or_create_moderator_thread(name: str) -> Optional[discord.Thread]:
    moderators_channel = bot.get_channel(MODERATOR_CHANNEL_ID)
    users_channel = bot.get_channel(USER_QUERY_CHANNEL_ID)
    if not isinstance(moderators_channel, discord.ForumChannel) or not isinstance(users_channel, discord.ForumChannel):
        logger.error("USER_QUERY_CHANNEL_ID and MODERATOR_CHANNEL_ID must both point to forum channels.")
        return None

    name = name[:100]
    try:
        if await find_thread(users_channel, name):
            logger.warning("A thread named '%s' also exists in the users-query channel.", name)

        existing = await find_thread(moderators_channel, name)
        if existing:
            return existing

        created = await moderators_channel.create_thread(
            name=name,
            content=(
                f"**New thread created for {name}** \n"
                "**-To get started, use the slash commands to access the guide and learn how to use this app**:\n\n"
            ),
            auto_archive_duration=10080,
        )
        await created.message.pin()
        logger.info("Created moderator thread '%s'.", name)
        return created.thread
    except discord.HTTPException:
        logger.exception("Failed to find or create moderator thread '%s'", name)
        return None


async def clean_thread(thread: discord.Thread) -> None:
    if not thread.permissions_for(thread.guild.me).manage_messages:
        logger.error("Bot lacks permission to manage messages in %s", thread.name)
        return

    try:
        to_delete = [
            msg async for msg in thread.history(limit=100)
            if not msg.pinned and not msg.content.startswith("/")
        ]
        now = discord.utils.utcnow()
        recent = [msg for msg in to_delete if (now - msg.created_at).days < 14]
        if recent:
            await thread.delete_messages(recent)
        # Bulk delete only works for messages younger than 14 days
        for msg in to_delete:
            if (now - msg.created_at).days >= 14:
                try:
                    await msg.delete()
                except discord.NotFound:
                    pass
    except discord.HTTPException:
        logger.exception("Failed to clean thread %s", thread.name)


COMBINE_PROMPT = (
    "Combine related information into one cohesive response, avoid repetition, "
    "and ensure proper formatting with appropriate spacing:\n\n"
)


class QueryQueue:
    def __init__(self):
        self.user_queries: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.moderator_context: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.processing: set = set()
        self.pending: set = set()

    async def add_user_query(self, context_key: str, query: str, message_id: int) -> None:
        self.user_queries[context_key].append((query, message_id))
        await update_query_display(context_key)

        if context_key in self.processing:
            self.pending.add(context_key)
            return
        self.processing.add(context_key)
        try:
            # Queries that arrive mid-generation trigger one more pass once the current one finishes
            while True:
                self.pending.discard(context_key)
                await send_comprehensive_response(context_key)
                if context_key not in self.pending:
                    break
        finally:
            self.processing.discard(context_key)

    async def add_moderator_query(self, context_key: str, query: str, message_id: int) -> None:
        self.moderator_context[context_key].append((query, message_id))
        await update_query_display(context_key)
        await send_comprehensive_response(context_key)

    async def remove_query(self, context_key: str, message_id: int) -> None:
        self.user_queries[context_key] = [(q, mid) for q, mid in self.user_queries[context_key] if mid != message_id]
        await update_query_display(context_key)

    async def reset(self, context_key: str) -> None:
        self.user_queries[context_key] = []
        self.moderator_context[context_key] = []
        await update_query_display(context_key)


query_queue = QueryQueue()


def format_queries(context_key: str, heading: str) -> str:
    author = original_author_mapping.get(context_key, "User")
    lines = [f"**{heading} {author}:**"]
    lines += [f"{number_to_emoji(i)} {q}" for i, (q, _) in enumerate(query_queue.user_queries[context_key], 1)]
    if query_queue.moderator_context[context_key]:
        lines += ["", "**Context from Moderator:**"]
        lines += [f"{number_to_emoji(i)} {q}" for i, (q, _) in enumerate(query_queue.moderator_context[context_key], 1)]
    return "\n".join(lines) + "\n"


async def update_query_display(context_key: str) -> None:
    mapping = thread_mapping.get(context_key)
    if not mapping or "moderator_thread" not in mapping:
        logger.error("Moderator thread not found for context_key: %s", context_key)
        return

    content = format_queries(context_key, "All Queries from")
    display_message = mapping.get("display_message")
    if display_message:
        try:
            await display_message.edit(content=content)
            return
        except discord.HTTPException as e:
            logger.warning("Could not edit display message for %s (%s), sending a new one.", context_key, e)
    mapping["display_message"] = await safe_send_message(mapping["moderator_thread"], content)


async def send_comprehensive_response(context_key: str) -> None:
    mapping = thread_mapping[context_key]
    thread = mapping["moderator_thread"]
    step = "Generating response"

    await clean_thread(thread)
    await spinner_manager.start_spinner(thread, SpinnerState.INITIALIZING)
    try:
        if not query_queue.user_queries[context_key] and not query_queue.moderator_context[context_key]:
            await safe_send_message(thread, "No queries or context found to respond to.")
            return

        content = format_queries(context_key, "Queries from")
        await spinner_manager.update_spinner_state(thread, SpinnerState.PROCESSING, step)
        response = await conversational_system.handle_query(
            context_key, f"{COMBINE_PROMPT}{content}\n\n", update_history=False
        )
        await spinner_manager.complete_step(thread, step)

        message = await safe_send_message(thread, f"{content}\n**Response**:\n\n{response}", view=ResponseView(context_key))
        if message:
            mapping["display_message"] = message
            mapping["response"] = response
    except Exception:
        logger.exception("Error sending response for context_key %s", context_key)
        await safe_send_message(thread, "An error occurred while generating the response.")
    finally:
        await spinner_manager.stop_spinner(thread)


async def process_moderator_chat_query(context_key: str, query: str) -> None:
    thread = thread_mapping[context_key]["moderator_thread"]
    await spinner_manager.start_spinner(thread, SpinnerState.PROCESSING)
    try:
        response = await conversational_system.handle_query(context_key, query, is_moderator_query=True)
        await safe_send_message(thread, f"**Response**:\n\n{response}")
    finally:
        await spinner_manager.stop_spinner(thread)


async def handle_new_message(message: discord.Message, edited: bool = False) -> None:
    channel = message.channel
    if not isinstance(channel, discord.Thread) or channel.id in muted_threads:
        return

    context_key = find_context_key(channel.id)
    if context_key and thread_mapping[context_key].get("muted"):
        return

    if channel.parent_id == USER_QUERY_CHANNEL_ID:
        context_key = f"{channel.id}_{message.author.id}"
        original_author_mapping[context_key] = message.author.mention
        if "moderator_thread" not in thread_mapping[context_key]:
            moderator_thread = await find_or_create_moderator_thread(f"{channel.name}_{message.author.name}")
            if moderator_thread is None:
                return
            thread_mapping[context_key].update(user_thread=channel, moderator_thread=moderator_thread, muted=False)
        await query_queue.add_user_query(context_key, message.content, message.id)

    elif channel.parent_id == MODERATOR_CHANNEL_ID:
        if context_key is None:
            # A thread the moderators started themselves: chat with the bot directly
            context_key = f"mod_{channel.id}"
            thread_mapping[context_key] = {"moderator_thread": channel, "user_thread": None, "muted": False}
            original_author_mapping[context_key] = message.author.mention

        if context_key.startswith("mod_"):
            await process_moderator_chat_query(context_key, message.content)
        else:
            if not edited:
                await clean_thread(channel)
            await query_queue.add_moderator_query(context_key, message.content, message.id)


class ContextModal(ui.Modal, title="Add Context"):
    context = ui.TextInput(label="Additional Context", style=discord.TextStyle.paragraph)

    def __init__(self, context_key: str):
        super().__init__()
        self.context_key = context_key

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        await query_queue.add_moderator_query(self.context_key, self.context.value, interaction.id)
        await interaction.followup.send("Context added successfully.", ephemeral=True)


class ResponseView(ui.View):
    def __init__(self, context_key: str):
        super().__init__(timeout=None)
        self.context_key = context_key

    @ui.button(label="Add Context", style=ButtonStyle.secondary)
    async def add_context(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.send_modal(ContextModal(self.context_key))

    @ui.button(label="Send to User", style=ButtonStyle.success)
    async def send_to_user(self, interaction: discord.Interaction, button: ui.Button):
        channel = interaction.channel
        await interaction.response.defer(ephemeral=True)

        mapping = thread_mapping.get(self.context_key, {})
        response = mapping.get("response")
        user_thread = mapping.get("user_thread")
        if not response:
            await interaction.followup.send("No recent responses found to send.", ephemeral=True)
            return
        if user_thread is None:
            await interaction.followup.send("Could not find the corresponding thread in users-query channel.", ephemeral=True)
            return

        await spinner_manager.start_spinner(channel, SpinnerState.SENDING)
        try:
            author_mention = original_author_mapping.get(self.context_key, "User")
            await safe_send_message(user_thread, f"{author_mention},\n\n{response}")
            confirmation = await channel.send(f"🚀 The response has been sent to the original thread: {user_thread.jump_url}\n")
            await confirmation.pin()
            await clean_thread(channel)
            await query_queue.reset(self.context_key)
        except discord.Forbidden:
            await interaction.followup.send("Error: Bot doesn't have permission to perform this action.", ephemeral=True)
        except discord.NotFound:
            await interaction.followup.send("Error: The user's thread was not found. It may have been deleted.", ephemeral=True)
        except discord.HTTPException as e:
            logger.exception("Failed to send response to user")
            await interaction.followup.send(f"Error: Failed due to a Discord API error: {e}", ephemeral=True)
        finally:
            await spinner_manager.stop_spinner(channel)


class GuideView(ui.View):
    def __init__(self):
        super().__init__(timeout=300)

    @ui.button(label="How to Use", style=ButtonStyle.primary)
    async def how_to_use(self, interaction: discord.Interaction, button: ui.Button):
        embed = discord.Embed(title="How to Use This Bot", color=discord.Color.blue())
        embed.add_field(name="1. Submit Queries", value="Post your questions in the #ask-for-help channel.", inline=False)
        embed.add_field(name="2. Bot Processing", value="The bot will collect and process your queries.", inline=False)
        embed.add_field(name="3. Moderator Review", value="Moderators receive queries and bot responses in a private channel.", inline=False)
        embed.add_field(name="4. Context Refinement", value="Moderators can provide additional context to improve responses.", inline=False)
        embed.add_field(name="5. Response Delivery", value="Moderators send approved responses back to users.", inline=False)
        embed.add_field(name="6. Query Reset", value="After moderator sends a response, the bot resets and awaits new queries.", inline=False)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @ui.button(label="Moderator Guide", style=ButtonStyle.secondary)
    async def moderator_guide(self, interaction: discord.Interaction, button: ui.Button):
        embed = discord.Embed(title="Moderator Guide", color=discord.Color.gold())
        embed.add_field(name="Accessing Queries", value="View user queries and bot responses in the #answer-bot channel.", inline=False)
        embed.add_field(name="Adding Context", value="Use `[Add context]` to refine the bot's understanding.", inline=False)
        embed.add_field(
            name="Sending Responses",
            value="Use `[send to user]` to approve and send the bot's response to the user.🚀 Only the response part will be sent when you use the 'Send to User' button.",
            inline=False,
        )
        embed.add_field(name="Query Reset", value="After sending a response, the bot will reset and await new queries from users.", inline=False)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @ui.select(
        placeholder="Select a topic",
        options=[
            SelectOption(label="Query Handling Process", value="process"),
            SelectOption(label="Moderator Context", value="context"),
            SelectOption(label="Response Cycle", value="cycle"),
        ],
    )
    async def select_topic(self, interaction: discord.Interaction, select: ui.Select):
        topics = {
            "process": "The bot collects user queries from a designated channel, processes them, and sends the queries along with comprehensive responses to a private moderator channel for review.",
            "context": "Moderators can provide additional context. This context is stored in a 'Context from Moderator' section and used by the bot to refine future responses.",
            "cycle": "After a moderator sends an approved bot response to the user, the bot resets its query collection process. It then begins collecting new queries from the user channel, starting a new cycle.",
        }
        await interaction.response.send_message(topics[select.values[0]], ephemeral=True)


@bot.tree.command(name="stop", description="Stops the bot from responding to further queries in this thread.")
async def stop_thread(interaction: discord.Interaction):
    thread = interaction.channel
    if not isinstance(thread, discord.Thread):
        await interaction.response.send_message("This command can only be used within a thread.", ephemeral=True)
        return

    muted_threads.add(thread.id)
    context_key = find_context_key(thread.id)
    if context_key:
        thread_mapping[context_key]["muted"] = True
        notice = "🚫 The bot will no longer respond to queries in this thread.\n"
    else:
        notice = (
            f"⚠️ This thread (ID: {thread.id}) was not found in the active conversations, "
            "but it has been muted to prevent further responses.\n"
        )
    await interaction.response.send_message(
        notice + "If you'd like to restart the bot in this thread, please use the **`/start`** command"
    )
    logger.info("Muted thread %s (ID: %s)", thread.name, thread.id)


@bot.tree.command(name="start", description="Allows the bot to start responding to queries in this thread again.")
async def start_thread(interaction: discord.Interaction):
    thread = interaction.channel
    if not isinstance(thread, discord.Thread):
        await interaction.response.send_message("This command can only be used within a thread.", ephemeral=True)
        return

    context_key = find_context_key(thread.id)
    was_muted = thread.id in muted_threads or bool(context_key and thread_mapping[context_key].get("muted"))
    muted_threads.discard(thread.id)
    if context_key:
        thread_mapping[context_key]["muted"] = False

    if was_muted:
        await interaction.response.send_message("✅ The bot will now respond to queries in this thread.")
        logger.info("Unmuted thread %s (ID: %s)", thread.name, thread.id)
    else:
        await interaction.response.send_message("The bot is already responding in this thread.", ephemeral=True)


@bot.tree.command(name="guide", description="Interactive guide on how to use this bot effectively")
async def guide(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🔍 Interactive Bot Guide",
        description="Welcome to the interactive guide! Use the buttons and dropdown below to learn more about the bot's features, query handling process, and moderator interactions.",
        color=discord.Color.blue(),
    )
    await interaction.response.send_message(embed=embed, view=GuideView())


@bot.event
async def on_ready():
    logger.info("%s has connected to Discord!", bot.user)


@bot.event
async def on_message(message: discord.Message):
    if message.author != bot.user:
        await handle_new_message(message)


@bot.event
async def on_message_edit(before: discord.Message, after: discord.Message):
    # Discord also fires edits when link previews load, so only react to real content changes
    if after.author != bot.user and before.content != after.content:
        await handle_new_message(after, edited=True)


@bot.event
async def on_message_delete(message: discord.Message):
    channel = message.channel
    if isinstance(channel, discord.Thread) and channel.parent_id == USER_QUERY_CHANNEL_ID:
        context_key = f"{channel.id}_{message.author.id}"
        if context_key in thread_mapping:
            await query_queue.remove_query(context_key, message.id)


def main() -> None:
    token = require_env("DISCORD_BOT_TOKEN")
    if "--reindex" in sys.argv or document_store.count_documents() == 0:
        ingest_documents(DOCS_DIR)
    query_pipeline.warm_up()
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
