# Memory Bridge: An Agentic Memory Layer for LLMs

This project is a hands-on assessment for the Senior ML Engineer role at Memory Bridge. It features the design and implementation of a lightweight, agentic memory system for Large Language Models (LLMs), complete with a robust benchmark suite for evaluation. The focus is on thoughtful system architecture, meaningful evaluation, and clear engineering judgment.

## Core Features
*   **Agentic Architecture:** The system is built around an intelligent agent (powered by `gemini-2.5-pro`) that uses a toolset to interact with a memory store, allowing for complex, multi-step reasoning.
*   **Structured Memory Store:** Utilizes ChromaDB to store memories as vector embeddings alongside rich metadata, including timestamps, confidence scores, and source information.
*   **LLM-Powered Memory Extraction:** The agent analyzes conversations to extract key facts, create new memories, and intelligently update existing ones.
*   **Automated Benchmark Suite:** A comprehensive evaluation pipeline (`benchmark/evaluate.py`) that tests the system's performance on a synthetically generated dataset.
*   **Comparative Analysis:** The agent's performance is compared against a simple keyword-based baseline to clearly demonstrate its strengths and weaknesses.

## System Architecture

The system operates on an agentic, tool-based architecture. This is a more robust and flexible approach than a simple linear pipeline.

1.  **Conversation Input:** A turn from the conversation is passed to the Agent.
2.  **Agent Analysis:** The Agent, powered by `gemini-2.5-pro` and a carefully engineered prompt, analyzes the conversation history. It decides if a memory operation is needed.
3.  **Tool Selection:** If an operation is needed, the Agent selects the appropriate tool from its toolbelt (`read_memory`, `write_memory`, `update_memory`) and generates the necessary arguments.
4.  **Tool Execution:** The selected Python function is executed. For example, `read_memory` generates an embedding and queries the `MemoryStore`.
5.  **Memory Store:** The `MemoryStore` (ChromaDB) performs the requested CRUD (Create, Read, Update, Delete) operation on the vector database.
6.  **Response to Agent:** The result of the tool execution is returned to the Agent, which can then decide on a subsequent action (e.g., calling `update_memory` after a successful `read_memory`).

## Project Structure
```
memory-bridge/
├── .env                  # Environment variables (API keys)
├── README.md             # This file
├── requirements.txt      # Project dependencies
├── main.py               # A simple script to demonstrate the agent
├── benchmark/
│   ├── dataset.json      # The synthetically generated dataset
│   ├── dataset_schema.py # Pydantic schema for the dataset
│   ├── generate_data.py  # Script to generate the dataset using an LLM
│   ├── evaluate.py       # Main script to run the full benchmark suite
│   ├── baseline.py       # Implementation of the simple keyword-based baseline
│   └── generation_prompts.md # The prompts used for data generation
└── src/
    ├── agent.py          # The core intelligent agent logic
    ├── data_models.py    # Pydantic models for Memory objects
    ├── memory_store.py   # Wrapper for the ChromaDB vector store
    └── tools.py          # The agent's toolset (read, write, update)
```

## Setup and Installation

### Prerequisites
*   Python 3.10+

### 1. Clone the Repository
```bash
git clone https://github.com/shakibaT/memory-bridge.git
cd memory-bridge
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
The project requires API credentials to connect to the LLM and Embedding endpoints.

1.  Create a file named `.env` in the root of the project directory.
2.  Add the following key-value pairs to the file, replacing the placeholder with your actual API key:

```env
OPENAI_API_KEY="your-api-key-from-the-email"
OPENAI_API_BASE="https://llm.internal.memorybridgeai.com/v1"
```
The database is created automatically in memory or on disk when the scripts are run; there is no separate database creation script needed.

## How to Run

### Running the Simple Demo
The `main.py` script runs a short, hardcoded conversation to provide a qualitative demonstration of the agent's ability to extract and update memories.

```bash
python main.py
```

**Expected Output:**
```
--- Initializing Memory Bridge Demo ---
INFO: MemoryStore initialized and connected to collection 'memories'.
INFO: Agent initialized with model 'gemini-2.5-pro' and 3 tools.

--- Starting Demo Conversation ---

--- Processing Turn 0 ---
AGENT: Executed 'write_memory'.
...
--- Processing Turn 1 ---
AGENT: Skipping turn because it is not a user message.

--- Processing Turn 2 ---
TOOL: Executing read_memory with query: 'user's current role'
AGENT: Executed 'read_memory'.
TOOL: Executing update_memory for fact 'fact_...'.
AGENT: Executed 'update_memory'.

--- Final State of Memory Store ---
ID: fact_..., Content: 'The user's name is Alice.'
ID: fact_..., Content: 'The user is a Director at Innovate Inc.'
...
--- System Shutdown ---
```

### Running the Benchmark Suite
This is a two-step process. First, you generate the synthetic dataset, then you run the evaluation pipeline.

**Step 1: Generate the Synthetic Dataset**
This script will use the LLM to generate ~5-10 conversations and save them to `benchmark/dataset.json`.

```bash
python -m benchmark.generate_data
```

**Step 2: Run the Evaluation**
This script will run both the advanced agent and the baseline system against the dataset and print a final performance report.

```bash
python -m benchmark.evaluate
```

**Expected Output:**
```
Loaded 5 conversations for evaluation.

--- Evaluating Advanced Agent System ---
Agent System: 100%|██████████| 5/5 [00:15<00:00,  3.01s/it]

--- Evaluating Baseline System ---
Baseline System: 100%|██████████| 5/5 [00:01<00:00,  3.35it/s]


--- Benchmark Results Summary ---
                               success_rate_%
system   metric
Agent    Overall Success Rate          85.71
Baseline Overall Success Rate          33.33
---------------------------------
This table shows the percentage of tasks each system completed successfully.
```

## AI Usage Notes

As encouraged by the project description, AI tools were used to accelerate development and focus on high-level design:
*   **Synthetic Data Generation:** `gemini-2.5-pro` was used in `benchmark/generate_data.py` to create realistic, structured conversation data for the benchmark suite. The prompts used are documented in `benchmark/generation_prompts.md`.
*   **Boilerplate Code:** ChatGPT/Claude were used to generate initial boilerplate for Pydantic models and basic class structures, which were then reviewed and adapted.
*   **Prompt Engineering:** The system prompt for the agent in `src/agent.py` was iteratively refined with the help of an AI assistant to improve its instruction-following capabilities.

## Limitations and Future Work
*   **Multi-Hop Reasoning:** The current agent can perform simple, implicit multi-hop reasoning but lacks a more sophisticated planning module. A future version could implement a ReAct-style framework for more complex queries.
*   **Error Handling:** The system's error handling is basic. A production system would require more robust error handling and retry mechanisms, especially for API calls.
*   **Memory Deduplication:** The agent relies on its prompt to avoid creating duplicate memories. A more advanced system would include a "pre-write" check that performs a similarity search to find and merge redundant facts.
*   **Privacy & Ethics:** This system is a proof-of-concept. A real-world deployment would require significant investment in privacy-preserving technologies, user consent flows, and an explicit, user-facing interface to view, edit, and delete their stored memories, in compliance with GDPR and other regulations.