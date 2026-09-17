# Discord AI Agent

A retrieval-augmented support bot for Discord. Ran 24/7 in Qdrant's Discord server, where it answered community questions from the Qdrant documentation. Moderators review each answer before it reaches the user.

Questions are matched against your docs with **hybrid search** (dense + sparse vectors in [Qdrant](https://qdrant.tech)), and **Claude** writes the answer from the passages it finds.

## How it works

```
User posts in the help forum
        │
        ▼
Bot opens a matching thread in the private moderator forum
        │
        ▼
Hybrid search over the indexed docs  ──►  Claude drafts an answer
        │
        ▼
Moderators review it in the thread
   ├─ [Add Context]  → extra context is added and the answer is regenerated
   └─ [Send to User] → the answer is posted in the user's thread, and the queue resets
```

- **Grouped queries:** follow-up messages from the same user go into one queue, and the bot answers them all together.
- **Direct chat for moderators:** a thread that a moderator starts in the moderator forum is a private chat with the bot, and the bot remembers the last few exchanges.
- **Live progress:** a status message shows elapsed time while an answer is being generated.
- **Slash commands:** `/guide` shows an interactive walkthrough, and `/stop` and `/start` mute or unmute the bot in a thread.

### Retrieval pipeline

1. Files in the docs folder are read with automatic encoding detection. The bot guesses each file's language (Python, JavaScript, Java, C#, Rust, HTTP, Bash or plain text) from its content.
2. Content is split at code boundaries such as functions, classes and imports. The pieces are then packed into chunks of up to 2,048 characters.
3. Each chunk is embedded with `snowflake/snowflake-arctic-embed-l` (dense, 1024-d) and `prithivida/Splade_PP_en_v1` (sparse), via [FastEmbed](https://github.com/qdrant/fastembed), and written to Qdrant with [Haystack](https://haystack.deepset.ai).
4. At query time, `QdrantHybridRetriever` merges the dense and sparse results.
5. If the question asks for code and the matched docs include code blocks, Claude gets a code-focused prompt. Otherwise it gets an explanation-focused prompt.

## Setup

### 1. Prerequisites

- Python 3.10+
- A Qdrant instance: a [Qdrant Cloud](https://cloud.qdrant.io) cluster, or a local one started with `docker run -p 6333:6333 qdrant/qdrant`
- An [Anthropic API key](https://console.anthropic.com)
- A Discord bot application with the **Message Content** intent enabled

### 2. Discord server

Create two **forum channels**:

| Channel | Purpose |
|---|---|
| User queries | Where members post their questions |
| Moderators | Private; the bot mirrors each question here for review |

Invite the bot with these permissions: View Channels, Send Messages, Send Messages in Threads, Create Public Threads, Manage Messages (for pinning and cleaning up threads), Read Message History, and Embed Links.

### 3. Install

```bash
git clone https://github.com/Ya-shh/Discord-AI-Agent.git
cd Discord-AI-Agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the values
```

### 4. Add your docs

Put the documentation files you want the bot to answer from in a `QDR/` folder. To use a different folder, set `DOCS_DIR`.

### 5. Run

```bash
python Discord-AI-agent.py            # indexes the docs on first run (when the collection is empty)
python Discord-AI-agent.py --reindex  # re-embeds and overwrites the docs, e.g. after updating them
```

The first run downloads the embedding models, which are about 1.5 GB.

## Configuration

| Variable | Required | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | yes | Discord bot token |
| `USER_QUERY_CHANNEL_ID` | yes | ID of the user query forum channel |
| `MODERATOR_CHANNEL_ID` | yes | ID of the moderator forum channel |
| `ANTHROPIC_API_KEY` | yes | Anthropic API key |
| `QDRANT_URL` | no | Qdrant URL (default `http://localhost:6333`) |
| `QDRANT_API_KEY` | no | Needed for Qdrant Cloud |
| `QDRANT_COLLECTION` | no | Collection name (default `disFY`) |
| `DOCS_DIR` | no | Folder to index (default `QDR`) |

The Claude model is set by `CLAUDE_MODEL` at the top of `Discord-AI-agent.py` (default `claude-opus-5`). Requests use Anthropic's server-side fallback, so if Claude declines a request, it's retried on Anthropic's recommended fallback model.

## Notes

- Thread links, queues and chat history are kept in memory, so they reset when the bot restarts. Indexed docs stay in Qdrant.
- Chunk IDs come from each chunk's content and source, so re-indexing overwrites existing points instead of duplicating them.

## Tech stack

[discord.py](https://discordpy.readthedocs.io) · [Haystack](https://haystack.deepset.ai) · [Qdrant](https://qdrant.tech) · [FastEmbed](https://github.com/qdrant/fastembed) · [Anthropic Claude](https://docs.anthropic.com)
