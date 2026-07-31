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
import logging

logger = logging.getLogger(__name__)

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

VALID_WORKFLOW_MODES = {"sequential", "graph", "supervisor"}


def create_dynamic_workflow(agent_configs, workflow_mode="sequential"):
    """
    Builds a LangGraph based on user-defined configurations.
    Supports 'sequential', 'graph', and 'supervisor' modes.
    """
    if not agent_configs:
        raise ValueError("At least one agent configuration is required.")

    if workflow_mode not in VALID_WORKFLOW_MODES:
        raise ValueError(
            f"Unsupported workflow_mode '{workflow_mode}'. "
            f"Choose from: {', '.join(sorted(VALID_WORKFLOW_MODES))}"
        )

    ids = [c["id"] for c in agent_configs]
    if len(ids) != len(set(ids)):
        raise ValueError("Agent ids must be unique within a workflow.")

    if workflow_mode == "supervisor" and len(agent_configs) < 2:
        raise ValueError("Supervisor mode requires at least one supervisor and one worker agent.")

    workflow = StateGraph(AgentState)
    node_ids = []
    
    # 1. Register Agents and KBs
    # In supervisor mode the first config is the supervisor (not a worker DynamicAgent).
    worker_configs = agent_configs[1:] if workflow_mode == "supervisor" else agent_configs

    for config in worker_configs:
        node_id = config['id']
        node_ids.append(node_id)
        
        # Resolve tools (silently skip unknown ids; warn for visibility)
        tools = []
        for t_id in config.get('tools', []):
            if t_id in TOOL_REGISTRY:
                tools.append(TOOL_REGISTRY[t_id])
            else:
                logger.warning(f"Unknown tool id '{t_id}' for agent '{node_id}' — skipped")
        
        # Register KB
        register_agent_kb(
            node_id, 
            config.get('knowledge_base') or '', 
            kb_type=config.get('kb_type') or 'markdown',
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
        
        workflow.add_node(node_id, agent.node_func)
        
    # 2. Define Connectivity Logic
    if workflow_mode == "sequential":
        # Linear flow: 1 -> 2 -> 3 -> END
        workflow.set_entry_point(node_ids[0])
        for i in range(len(node_ids) - 1):
            workflow.add_edge(node_ids[i], node_ids[i+1])
        workflow.add_edge(node_ids[-1], END)
        
    elif workflow_mode == "graph":
        # Explicit Graph Routing (parallel fan-out supported)
        workflow.set_entry_point(node_ids[0])
        for config in agent_configs:
            uid = config['id']
            next_steps = config.get('downstream_nodes', []) or []
            valid_targets = [t for t in next_steps if t in node_ids and t != uid]

            if valid_targets:
                for target in valid_targets:
                    workflow.add_edge(uid, target)
            else:
                # No valid next steps → terminate this branch
                workflow.add_edge(uid, END)

    elif workflow_mode == "supervisor":
        # First agent is the Supervisor; remaining agents are workers.
        supervisor_config = agent_configs[0]
        worker_ids = node_ids  # already only workers from above

        supervisor = SupervisorAgent(
            supervisor_config['name'], 
            supervisor_config['system_prompt'], 
            worker_ids,
        )
        workflow.add_node("supervisor_router", supervisor.node_func)
        workflow.set_entry_point("supervisor_router")
        
        members = {cid: cid for cid in worker_ids}
        members["FINISH"] = END
        
        workflow.add_conditional_edges(
            "supervisor_router",
            lambda x: x.get("next") or "FINISH",
            members
        )
        
        for wid in worker_ids:
            workflow.add_edge(wid, "supervisor_router")
            
    # Cap recursion to prevent infinite supervisor / tool loops from hanging the API
    return workflow.compile()
