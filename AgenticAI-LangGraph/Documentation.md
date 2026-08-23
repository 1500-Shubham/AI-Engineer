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
- e) Evaluator Optimizer: Task-> cant execute perfectly in single go, its like iteration in steps Rejected+Feedback like creative iteration Generator and Evaluator LLM : solution plus feedback loop imporvement -> final accept or reject with feedback

2) Graph, Nodes and Edges
- All high goals -> actionable steps (generate topic -> collect essaay -> evaluate -> aggregate -> conditional -> feebdack ) -> Represent in Graph form, all conditions, human in the loop, etc

3) State: All data points needed in complete workflow, help in gudining the LLM (score, topic, feeback, etc) OBJECT {store all for graph}, accessibke to all the nodes in graph, mutable change allowed

4) Reducers: State are accessible to all nodes and mutable, Reducers tells how nodes upadte the state (upadte,merge,add) -> each key in state has its own reducers
    - Lets say workflow (a,b) -> sum -> square
    - State {firstNo, secondNo, result} 5,6 -> 11 -> 121 
    - Problem: update result field can be problem, like in chatbot field message: it keeps on updating, but we want to keep all messages for Chatbot

5) Execution Model of LangGraph: Inspired by Google Prygel- system large scale graph processing
a) Graph Definiton - workflow behalf - node edges and state
b) Compilation -> .compile on the StateGraph - check graph structure- orphan nodes etc
c) Invocation -> .invoke(inital_state) to first node
d) Super-Steps Begin : Execution proceeds in rounds: Parallel steps invocation 
e)  Messagae PAssing and Node Activation
f) Halting Condition : No more nodes and condition fulfilled

## LangGraph Workflows: Lec 6:9
### Sequential Workflows: VERY EASY
- Workflow -> add nodes (fucntion / llm call), edges, Functions takes State(Exact Dictionary) as input and output see code in series
- Prompt Chaining -> START -> generate Outline of topic -> Generate Blog -> END

### Parallel Workflows: 
- Simple Parallel workflow - here when 3 parallel function return same State (taking same state input) -> kind of conflict completeSTATE returned by all three nodes -> Langgraph think all state changed: CONFLICT
    - Input and output excpect Dictionary only -> staet management happened inside function only: update wagera -> only return dictionary ho same state nahi ya same key conflict
- Parallezation Workflow -> LLM Based : REDUCERS
    - UPSC essay -> a) COT- clarity of thought b)DOA c) Language all abc has a score so need a State with 3 fieds
    - Use Reducer Function -> individual_score field [a b c] need to do merging here: by default field replace/update: need reducer function
    - Model with structured output will use based on schema defined - Langchain BaseModel  BaseModel -> Field Description which LLM understands
    - Now continuing with State of Langgraph concept easy mixing
    - state: individual_score: list[int] -add Reducer here : Annotate[list[int],operator.add] -> [5] + [6] + [7] -> operator is doing addition and telling to add, returning from functuion we do {a:a, individual_score:[5]} like this all returned there value and the reducer keeps on combining the state value -> next node merge happened to that field in state
    - sum(individual_score)/len(individual_score) can be used

### Conditional Workflows:
-  Instead of parallel we add condiiton to select one instead of many -> 1 to 2 or 1 to 3
- Non-LLM based: Quadratic Equation Workflow: 
    - *Condition function create: whose output is other function names : if state[a]>0 return function name which is being returned
    - Routing function act, tell the node name
    - Instead of add edgees we do conditional edges with a router function
    - do like graph.add_conditional_edge("nodeName",router function which return the node name) -> so edge converted based on condition
- LLM Based: Review Handling Workflow : Customer review reply sentiment positive or negative then reply write
    - Postiive negativfe -> strucutred output use BaseModel & Field -> Langchain type
    - Reply -> LLM help prompt : lets say Diagnosis also want in strcutred {dict form output} then define in BaseModel
- Another way to route to a node like End and another node based on condtion function lets say condition function returned {a,b} json then while adding condition edge {node, fucntion , {a->END, b->Antoher node}} : just syntax type

### Iterative Workflows:
- a -> needs improvement -> c and then want to go back to c(optimize stage) -> a(evaluate)
- can have iteration , max iteration count here or end, storing history in state using operator
- also while prompting can use, System and user messages -> langchain


## Basic Chatbot using LangGraph: Lec 10
- start -> chat_node -> end (state needed: messages [operator added -> BASEMESSAGE -> (langgraph add_message use) user and llm]) -> Can use BASEMESSAGE, HUMANMESSAGE, AIMESSAGE, SYSTEMMESSAGE types of messages storing : Base Message is parent of everything
- intial_state: messages: [HumanMessage(content='')] aise karke bhejoge and reponse = llm.invoke store again in messages AIMESSAGE (contnet= ) sotred automatically
- while True: lagake call llm graph invoke -> user input contains 'Exit' type check: Problem is inside WHILE loop we are invoking Graph (start -> end) previous invocation deleted the new state
- Problem: loop -> invoke call -> state reseted -> flaw
    - Solution: Persistence in Langgraph -> workflow trigger using invoke (state persisted in memory) -> Local RAM -> MemorySaver Checkpointer Memory
    - checkpointer = Memory Saver() -> graph.compile (checkpointer give the db storing), Also while invoking chatbot -> pas some config={if same then use same db context saved} like key - values getting stored
    - if we do chatbot.getState(same Config) -> return all vlaues stored in persistence memory config={some vlaue aacting as key}

## Persistence in LangGraph | Time Travel
- store and restore state of workflow -> since all nodes in graph share same state
- state: {final_value, intermediate value} -> db store -> fault tolerance help if any node crash we know the state values
### Checkpointers : State values keep on saving when graph traversal
- superstep -> checkpoint convert parent:child relation
- Theads/ Config in Persistence: like using same graph with diffferent key value pair 
- chatbot.getStaet(config) pass and get
### Implementation Code: Joke -> explanation generate -> end
- Step:1- In Memory Saver(): checkpointer= InMemorySaver() and simply worflow = graph.compile(checkpointer=checkpointer)
- Step:2- config1={"configurable":{"thread_id:"1} : workflow.invoke({state value start},config=config)
- Step-3: workflow.get_state(config) // get final value of state based on config
- Step-4 worflow.get_state_history()
### Benefits of Persistence

a) Short Term Memory b) Fault Tolerance c) Human in the loop d) Time Travel

#### Fault Tolerance-> crash at some node -> resume from that node
- Mimic try-one node -> delay 30seconds - manuallys- keyboard stop
- Notice that it will start from next node -> use graph.get_state()
- Resume: again call graph.invoke(None,{same config}) easyyy

#### Human in the loop:
- Workflow -> topic->Linekdin Post-> Ask my permission(HITL) -> API call : By default LangGraph do interrupt before human in the loop
- A dedicated section future

#### Time Travel: Workflow execution replay after its finished
- Can go to a particular node and resume from that and replay further steps
- Helps in debugging as going to that checkpoint and resume again
###### Replay the workflow from some state
- Steps: use get_state_history and get tthat checkpoint reaching there
   - Step1 - each checkpoint has its ID
   - workflow.get_state("config=thread1, "checkpoint_id":211)
   - workflow.invoke(config, checkpoint_id="") from this state/checkpoint resume the workflow and replay state changed
   - State History: now has all previous + new state information
###### Update state using checkpoint 
- workflow.update_state(config,checkpointid, new_vlaue_state at that place)
- worflow.invoke(None, config,checkpoint) // replay from that state

## Langsmith: Observability in LLM Applications

## Tools
- Chatbot add : Numerical Calculation, Internet Search, Stock Tool (Company stock price at this moment)
a) Fundamentals
- chat_node -> decision making -> Tool call or chatting deciside : CHAT_NODE DECSION OR ACTION -> Tool NODE
- All Tools Collection -> Special TOOL NODE (handle all tools) -> Chat node send tool name -> TOOL NODE -> exectute that tool with tool name and query to be used
- Tool node: bridge between graph and external Tools (function API utilities)
- Tools Condition: Question based chatting or tool call needed -> prebuild condiitonal edge function

b) Code Use:
- ToolNode, tool_condition from langgraph.prebuild
- Langchain -> DuckDuckGo -> tool (custom tool create)
- Step:1 Create all Tools (Prebuild Tool and Custom Tool)
    - search tool - prebuild 
    - @tool function def(params) -> dict -> info about it comment docString needed LLM read 
    - @tool -> def (var) -> dict -> comments and API 
- Step2: 
    - tools = []
    - model.bind_tools(tools) // upto this langchain llm ready with tools
- Step3: LangGraph flow: support
    - Need to create ChatNode and ToolNode toolNode-> (tools=[] input give)
    - ChatNode  mode_with_tool.invoke()
    - EDGES: graph.add_condition_edge(chatnode, tools_conditions) -> here Langgraph direct to Tool Node if needed
- Problem -> ChatNode -> ToolNode (output dict) -> return as END: Need LLM after ToolNode -> ChatNode then end refined way , ALSO two way two Tools -> Tool1 -> LLM -> Tool2 -> LLM -> Refined way
- Solution: 
    - graph.add_edge("tools","chat_node")
    - tools_condition output -> TOOL NODE or END NODE

c) Integrate this tool concept inside CHATBOT

## MCP Client: Tools live in Server having all TOOLS:
a) Fundamentals -
- improved version of tools : better way to standarize way to connect to llm applications. 
- Normal Tool Flaw -> ex. get github prs function -> pull request show
    - Approach is brittle: can break no gurantee works tomorrow,
    - Chatbot(Tool calling API) -> Github (API) : Github updated the APIv.2 payload params, better use MCP Client of Github directly, we have to change code for each Client
- MCP: Bridge build Chatbot (Client) only config which dont change like token -> Github (Server) : github maintains only
- SERVERS = {config code connect to github } -> tools and its definition which can pe passed to LLM easy - solved maintenance code
- Connect to URL MCP - {transport: , url:}

b) Build own MCP Server with all tools -> FASTMCP
- Need to watch playlist MCP Proper Playlist
- Basic:
    - mcp = FastMCP
    - @mcp.tool def fucntion with async -> comment for llm

c) Basic Code:
- Replacing Tool with MCP Client and Server
- ##### MCP Client needs async code -> convert into async
    - ASYNC Convert: chatbot.ainvoke -> await use
    - asyncio.run(main()) -> async def main():
    - make all tools ayncs using ayncc def and ainvoke
- ###### MCP Client Build : Replace tools with Client
    - from langchain_mcp_adpater.cleint import MULTIServerMCPClient
    - client decalare = {name:{args etc command}}
    - can connect multiple mcp in a single go same client = multiple configs
    - to run mcp server we write a command in config only- command python +argscombinatron something
    - Client only starts the MCP Server with config
    -
- ##### Connect Client to Graph
    - tools = await client.get_tools() -> async function
    - print tools
    - llm_with_tools = llm.bind(tools)
    - SAME as Tools
- #####
d) Integrate in Chatbots: Pending

## RAG
a) Fundamentals: 
- Need data privacy or personal knowledge
- Query + Context(my rag) -> Prompt -> LLM
- Architecture -> 
    - Knowledge Source -> Split Smaller Parts -> Embedding model / vector -> store in db -> semantic similarity
- LangGraph -> TOOL as define in RAG* Template use noice

b) Code Learning: Rag Code
- i) Loader PDF library (intro-to-ml.pdf) 
- ii) splitter = RecursiveCharacterTextSplitter (chunk size, overlap), all chunks got
- iii) embedding model = OpenAIEmbeddigns
- iv) vector_store = FAISS all chunks and embeddings -> got all vectors //local db keh sakte stored
- v) Create Retriever-> searching: similarity with k:4 top4 similar : retriever = vector_store.as_retriever

c) LangGraph integrating RAG
- i) @tool rag_tool def (query): "MESSAGE" -> retriever.invoke(query) -> relevant result has document Object:{ id, metadata, page_content-realvalue}
    - return {content, metadata, query} -> use by LLM as context
- ii) tools = [rag_tool]
llm_with_tools = llm.bind_tool(tools)

-iii) Langgraph -> concept start -> state, graph, node, edges, invoke

c) Chatbot integrate : Pending

## Human in the loop: HITL
- supervise, approve, correct or guide model's output
- ensures -> ethical, safety , accountability

## SubGraphs
