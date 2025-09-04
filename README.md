# Multi-Agent Programming Assistant Chatbot

## Agent Flow Diagram

```
┌─────────────────┐
│   User Query    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   RouterAgent   │
└─────────┬───────┘
          │
    ┌─────┴──────────────────┐
    │                        │
    ▼                        ▼
┌──────────┐        ┌────────────────┐
│ Research │        │  Conversation  │
│  Agent   │        │     Agent      │
└───┬──────┘        └────────┬───────┘
    │                        │
  ┌─┴──────────┐             │
  │            │             │
  │            │             │
  │            │             │
  ▼            │             │
┌─────────┐    │             │
│  Code   │    │             │
│  Agent  │    │             │
└────┬────┘    │             │
     │         │             │
     │         │             │
     ▼         ▼             ▼
┌───────────────────────────────┐
│                               |
|         Presenter Agent       │
│                               |
└───────────────────────────────┘

```

## Agents Description

### 1. Router Agent

**Purpose**: Entry point that analyzes user queries and routes them to appropriate specialized agents.

**Tools**:

- `ClassifierTool`: Determines query intent (DOCUMENTATION, CODE, GENERAL)
- `DelegationTool`: Routes queries to appropriate agents based on classification

**Routing Logic**:

- DOCUMENTATION queries → ResearchAgent
- CODE queries → ResearchAgent → CodeAgent
- GENERAL queries → ConversationAgent

### 2. Research Agent

**Purpose**: Specializes in finding and filtering relevant documentation using RAG (Retrieval-Augmented Generation).

**Tools**: Uses RAG Manager directly for vector database search

**Knowledge Base**: 146+ LangChain documentation PDFs processed into 1516 chunks with embeddings

### 3. Code Agent

**Purpose**: Generates practical code solutions based on research findings and user requirements.

**Tools**:

- `CodeGeneratorTool`: Creates code examples and implementations
- `LinterTool`: Validates and formats generated code for correctness

**Capabilities**: Generates Python code with proper formatting, error handling, and best practices

### 4. Conversation Agent

**Purpose**: Handles general conversations, small-talk, and conceptual questions without requiring technical documentation.

**Approach**: Uses direct LLM calls for natural, engaging conversations without predefined responses or complex tools.

### 5. Presenter Agent

**Purpose**: Final processing node that formats and enhances all responses for optimal presentation.

**Tools**:

- `TextFormatterTool`: Applies markdown formatting, code block styling, and visual enhancements

**Quality Control**: Assesses presentation quality and ensures consistent formatting standards

## Agent Interaction Paths

### Path 1: Documentation Query (DOCUMENTATION Intent)

```text
User → RouterAgent → ResearchAgent → PresenterAgent → Response
```

### Path 2: Code Generation Query (CODE Intent)

```text
User → RouterAgent → ResearchAgent → CodeAgent → PresenterAgent → Response
```

### Path 3: General Conversation (GENERAL Intent)

```text
User → RouterAgent → ConversationAgent → PresenterAgent → Response
```

**Key Decision Points:**

- RouterAgent determines intent and routes to Research or Conversation
- After ResearchAgent, the system decides: if CODE intent → CodeAgent, else → PresenterAgent
- All paths converge at PresenterAgent for final formatting
