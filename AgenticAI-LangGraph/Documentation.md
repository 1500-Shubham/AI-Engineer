# Agentic AI Using LangGraph

## Theories Lec 1-4
- Agentic AI - Combination of LLM brain, Tools (APIs MCP), Memory, Conditions to use tool or return response
    - Autonomous, Goal Oriented, Planning, Reasoning, Adaptability, Context Awareness (Short term and Long term Memory)
- Gen AI - Generate content using LLM brain, can be multimodal
- LangChain vs LangGraph 
    - Langchain - Need modular building blocks to build workflows 
        - Models (Any LLM Provider can be called unified interface), Prompt, Retrievers (fetch documents from vector Store or Knowledge Base) and Biggest Offering Chains (Prompt -> Model -> Output Parser i/p and o/p of each other automatically) All Abstraction synced no manual coding needed
        - everything is component here tie and use
        - Conversation Workflow : Chatbots, Text Summarizers
        - Multistep Workflows (Prompt -> LLM -> Prompt -> LLM -> Parser)
        - RAG Application (Prompt -> Retriever:VectorStore -> LLM -> Response)
        - Basic Level Agents (Tools we have LLM can decide which Tools to call) : Weather API tool
    - Workflow vs Agents
        - Workflows are systems where LLMs and tools are orchasterated thorugh predefined code paths
        - Agetns LLM dynamically direct their own process and tool usage, maintaining controle over how they accomplish tasks.
        - Worflow STATIC is predefined in same order made by developer. AGENTS can have dynamic workflow build by agents depending on situation 
    - Will try a complex workflow in Langchain and problem ex. Automated Hiring Workflow
        - Problems: while implementing in Langchain : Static workflow problem not dynamic -> need this intelligence and flexibilty
    - Challenge1: Non Linear
            - Conditional Branch
            - Loops and Jumps
        - GLUE CODE-> Because of running loop, need python code to stich the chains manually
        - Langchain dont have construct for loops, jump, condition -> need python code to glue it\
        - LangGraph Advantage
            - Graph and edges : Conditional Edges: Loop back components
            - Beauty: Custom Python Zero GLUE CODE: no while if else loop
            - Complex workflow in graph form works perfect: Loops and branching
    - Challenge2: Handling State:
        - Data Points: JD, approved, post, job apply, all are data points needed for LLM and function to make decision
        - state = {goal, jd, num_applciation} this is like data point SET AND GET
        - Langchain Stateless -> memory (but conversation memory from LLM only supported)
            - but json type state not possible need to declare manually dictionary -> GET AND SET
        - LangGraph -> Stateful : Graph Create -> State Object -> all nodes of graph has this state object read and write
    - Challenge 3: Event Driven Execution : Langchain only works sequential once start cant stop in between (Manually first chain works 7 days wait python code then start second chain plus state transfer)
        - Worflow run (sequential or event driven)
        - Sequential Flow: Execution flow without stopping the stages
        - Event Driven -> Same stages where moving forward pause and wait for a Trigger like stop for 7 days to move next, WAIT 48 hours resume from same place
        - LangGraph: Inherit Even Driver: Node reach store state Checkpoints then pause for external trigger
    - Challenge 4: Fault tolerance (System able to recover when error happened in system, long running workflow can have issue) example: API error, server down where workflow is defined
        - LangChain: Dont have fault tolerance: restart the execution from start again :( Short lived Chain
        - LangGraph: Built In Fault Tolerance
            - Retry: for small errors wait and retry again after small time API Down support available
            - Big Fault: Server Down: Recovery Concept - Exacttly the place which node is failed Check Pointer concpet along with StateFull (In memory or Databases memory)
            - Each node exectutiion result stored as check points: Graph resume previous state
    - Challenge5: Human in the loop:
        - Langchain: no default mechanism to stop the chain, only way in steps of workflow in between ask for some input: problem will be the workflow is stuck in that flow consuming resources. Alternate create two chains after first chain-> then after human approval -> second chain continue (state of first need to pass to second)
        - LangGraph- First Class citizen for human in the loop: graph states and context, asynchorous human review allowed, checkpointer progress saved
    - Challenge 6: Nested Workflow- replace a single node with a new graph (sub-workflows useful MULTIAGENT SYSTEM)
        - example self driving car: multiple agent -> media, sensor, combine together complex problem solve
        - Reusability: can use subgraph as reusable in other places, like function in programming  
    - Challenge7: Observability: Able to debug understand, debug, workflow at runtime: LangSmith integrate in langchain monitor - chain steps llm calling record with prompt request response token time
        - LangSmith can not monitor GLUE CODE (in case of Langchain ex. looping)
        - Thats why use LangGraph besttt
- Langraph handles workflow orchestration while Langchain providng building blocks for each step in workflow.

## LangGraph Core Concepts: Lec 5
- Orchaestration framework integlligent, stateful, multi step workflows
- Parallelism loop, branching, memory and resumability
- Graph of nodes (task) and edges (routing): non- linear

1) LLM Workflows
- step by step process, each step is distinct task (prompting, reasoning, tool calling, memory access, decision making) - can be linear, branched, looped : retries, multi agents communication
- a) Prompt chaining - Gate (Checks) , Out, Exit
- b) Routing - LLM Call Router decision maker -> (who will execute LLM Call1, LLM Call2, etc.) QUERY Based decide router
- c) Parallelization -> Task break down into subtask -> merge result in Aggregator. Ex. You tube content moderation platform -> same video multiple angle check (community LLM1 guideline, misinformation LLM2, sexual content LLM3)
- d) Orchestrator Workers: similar to parallel but Task-> parallel subtask (nature of subtask not known dynamically LLM behaviour) ex. Research query -> LLM Call1 (google scholar) LLM Call2 (resarch)
Query Comes -> Orchastrator now decide to send to whom to, input query task nature dont know
- e) Evaluator Optimizer: Task-> cant execute perfectly in single go, 