from langgraph.graph import StateGraph, END
from stm.state import AgentState
from agents.dynamic_agent import DynamicAgent
from agents.supervisor import SupervisorAgent
from tools.financial_tools import (
    parse_financial_document,
    calculate_cagr,
    calculate_sip_future_value,
    calculate_stock_returns,
    get_mutual_fund_info,
    search_knowledge_base
)
from tools.integration_tools import (
    trigger_external_workflow,
    send_gmail_message,
    create_calendar_event,
    send_discord_message,
    call_marketplace_connector
)
from utils.knowledge import register_agent_kb

# Registry of available tools that UI can request
TOOL_REGISTRY = {
    "financial_parser": parse_financial_document,
    "cagr_calculator": calculate_cagr,
    "sip_calculator": calculate_sip_future_value,
    "stock_returns": calculate_stock_returns,
    "mf_info": get_mutual_fund_info,
    "kb_search": search_knowledge_base,
    "n8n_integration": trigger_external_workflow,
    "gmail_send": send_gmail_message,
    "calendar_event": create_calendar_event,
    "discord_send": send_discord_message,
    "marketplace_connector": call_marketplace_connector
}

def create_dynamic_workflow(agent_configs, workflow_mode="sequential"):
    """
    Builds a LangGraph based on user-defined configurations.
    Supports 'sequential' and 'supervisor' modes.
    """
    workflow = StateGraph(AgentState)
    node_ids = []
    
    # 1. Register Agents and KBs
    for config in agent_configs:
        node_id = config['id']
        node_ids.append(node_id)
        
        # Resolve tools
        tools = [TOOL_REGISTRY[t_id] for t_id in config.get('tools', []) if t_id in TOOL_REGISTRY]
        
        # Register KB
        register_agent_kb(
            node_id, 
            config.get('knowledge_base', ''), 
            kb_type=config.get('kb_type', 'markdown'),
            kb_provider=config.get('kb_provider'),
            kb_url=config.get('kb_url'),
            kb_api_key=config.get('kb_api_key'),
            kb_embedding_model=config.get('kb_embedding_model')
        )
        
        # Instantiate agent
        agent = DynamicAgent(
            config['name'], 
            config['system_prompt'], 
            tools,
            temperature=config.get('temperature', 0),
            guardrails=config.get('guardrails'),
            agent_id=node_id,
            webhook_url=config.get('webhook_url')
        )
        
        # Add to graph
        workflow.add_node(node_id, agent.node_func)
        
    # 2. Define Connectivity Logic
    if not node_ids:
        workflow.set_entry_point("__end__")
    
    elif workflow_mode == "sequential":
        # Linear flow: 1 -> 2 -> 3 -> END
        workflow.set_entry_point(node_ids[0])
        for i in range(len(node_ids) - 1):
            workflow.add_edge(node_ids[i], node_ids[i+1])
        workflow.add_edge(node_ids[-1], END)
        
    elif workflow_mode == "graph":
        # Explicit Graph Routing (Parallel supported here!)
        workflow.set_entry_point(node_ids[0])
        for config in agent_configs:
            uid = config['id']
            next_steps = config.get('downstream_nodes', [])
            
            if next_steps:
                for target in next_steps:
                    if target in node_ids:
                        workflow.add_edge(uid, target)
            else:
                # If no next steps, go to END
                workflow.add_edge(uid, END)

    elif workflow_mode == "supervisor":
        # Supervisor flow:
        # The first agent is designated as the Supervisor.
        # It decides which worker to call.
        supervisor_id = node_ids[0]
        worker_ids = node_ids[1:]
        
        # Re-register the first node as a Supervisor node
        # (It replaces the DynamicAgent logic for the first node in supervisor mode)
        supervisor_config = agent_configs[0]
        supervisor = SupervisorAgent(
            supervisor_config['name'], 
            supervisor_config['system_prompt'], 
            [agent_configs[i]['id'] for i in range(1, len(agent_configs))]
        )
        workflow.add_node("supervisor_router", supervisor.node_func)
        
        # Entry point is the router
        workflow.set_entry_point("supervisor_router")
        
        # Map of worker names/ids to graph nodes
        members = {cid: cid for cid in worker_ids}
        members["FINISH"] = END
        
        # Add conditional edges from supervisor to workers
        workflow.add_conditional_edges(
            "supervisor_router",
            lambda x: x["next"],
            members
        )
        
        # Add edges from workers back to supervisor
        for wid in worker_ids:
            workflow.add_edge(wid, "supervisor_router")
            
    return workflow.compile()
