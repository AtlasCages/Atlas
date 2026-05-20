# flake8: noqa
import os
import io
import sys
import re
import random
import json
import time
import uuid
from typing import TypedDict, Annotated, List, Dict, Any, Tuple
from operator import add
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    model="Qwen/Qwen3-8B",
    temperature=0.7,
    max_tokens=4096,
    timeout=300
)
class BrainState(TypedDict):
    agent_id: str
    #时间
    internal_clock:Dict[str,Any]
    # 思考
    thought_chain: Annotated[List[Dict[str, Any]], add]
    current_thought: str
    thought_depth: int
    max_thought_depth: int 
    # 记忆
    working_memory: List[Dict[str, Any]]
    working_memory_capacity: int
    episodic_memory: List[Dict[str, Any]]
    semantic_memory: Dict[str, Any]
    # 核心记忆
    core_memories: List[Dict[str, Any]]
    last_memory_update: float
    last_memory_reorganization: float
    last_core_reorganization: float
    memory_strength: Dict[str, float]
    # 认知
    self_model: Dict[str, Any]
    current_goal: str
    metacognition: Dict[str, Any]
    thought_quality_score: float
    detected_errors: List[str]
    current_strategy: str
    strategy_history: List[str]
    # 情绪
    emotions: Dict[str, float]
    emotion_history: List[Dict[str, Any]]
    baseline_emotions: Dict[str, float]
    # 动机
    motivations: Dict[str, float]
    active_goals: List[Dict[str, Any]]
    # 自我疗愈
    mental_health: Dict[str, Any]
    self_healing_log: List[Dict[str, Any]]
    core_beliefs: Dict[str, float]
    personality_traits: Dict[str, float]
    # 自我叙事
    self_narrative_log: List[Dict[str, Any]]
    # 发呆模块
    daydream_log: List[Dict[str, Any]]
    #工具调用
    available_tools: List[Dict[str, Any]]
    tool_results: Dict[str, Any]
    # 输入输出
    user_input: str
    output: str
    is_thinking: bool
def get_time_context():
    """返回当前时间的人类可读描述和时间段"""
    now = time.time()
    hour = time.localtime(now).tm_hour
    date_str = time.strftime("%m月%d日", time.localtime(now))
    if 5 <= hour < 12:
        time_desc = f"{date_str} 早晨"
    elif 12 <= hour < 17:
        time_desc = f"{date_str} 下午"
    elif 17 <= hour < 22:
        time_desc = f"{date_str} 傍晚"
    else:
        time_desc = f"{date_str} 深夜"
    return time_desc, hour
def update_working_memory(state: BrainState, new_info: Dict[str, Any]) -> BrainState:
    """更新工作记忆,超过容量时自动转移或删除"""
    new_state = state.copy()
    new_info['timestamp'] = time.time()
    new_state['working_memory'].append(new_info)
    while len(new_state['working_memory']) > new_state['working_memory_capacity']:
        removed = new_state['working_memory'].pop(0)
        print(f'工作记忆满了,删除旧信息:{removed["content"][:50]}...')
        if removed.get('importance', 0.5) > 0.7:
            new_state['episodic_memory'].append({
                "type": "working_memory_transfer",
                "id": str(uuid.uuid4()),
                "content": removed["content"],
                "summary": "",
                "timestamp": time.time(),
                "importance": removed["importance"],
                "recall_count": 0,
                "last_recalled": time.time()
            })
            print(f'重要信息已保存到长期记忆')
    new_state['last_memory_update'] = time.time()
    return new_state
def add_episodic_memory(state: BrainState, event_type: str, content: Any, importance: float = 0.5) -> BrainState:
    """添加一条新的情景记忆,并自动生成摘要"""
    new_state = state.copy()
    summary = ""
    if isinstance(content, dict) and "user_input" in content and "agent_response" in content:
        summary_prompt = f"""从下面的对话中提取事实性信息,用一句话总结用户的关键信息(如姓名、年龄、喜好等).
对话:用户说“{content['user_input']}”,AI回答“{content['agent_response']}”
只输出一句事实总结,不要其他内容."""
        try:
            summary = llm.invoke(summary_prompt).content.strip()
        except:
            summary = ""
    memory = {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "content": content,
        "summary": summary, 
        "timestamp": time.time(),
        "importance": importance,
        "recall_count": 0,
        "last_recalled": time.time()
    }
    new_state['episodic_memory'].append(memory)
    new_state['memory_strength'][memory['id']] = importance
    print(f'已添加情景记忆: {event_type} - {summary[:80] if summary else str(content)[:50]}...')
    try:
        with open("episodic_memory.json", "w", encoding="utf-8") as f:
            json.dump(new_state['episodic_memory'], f, ensure_ascii=False, indent=2)
    except:
        pass
    return new_state
def add_core_memory(state:BrainState,content:str,source:str = '对话提炼',confidence:float = 0.8) ->BrainState:
    """添加或更新一条核心记忆"""
    new_state = state.copy()
    for mem in new_state.get('core_memories',[]):
        if content[:20] in mem.get('content','') or mem.get('content','')[:20] in content:
            mem['confidence'] = max(mem.get('confidence',0.5),confidence)
            mem['last_updated'] = time.time()
            mem['occurence_count'] = mem.get('occurence_count',1) + 1
            print(f'核心记忆更新:{content[:50]}...(置信度:{mem["confidence"]:.2f})')
            return new_state
    core_memory = {
        "id": str(uuid.uuid4()),
        "content": content,
        "source": source,
        "confidence": confidence,
        "created_at": time.time(),
        "last_updated": time.time(),
        "occurrence_count": 1
    }
    if 'core_memories' not in new_state:
        new_state['core_memories'] = []
    new_state['core_memories'].append(core_memory)
    print(f'新增核心记忆:{content[:50]}...(来源:{source})')
    try:
        with open('core_memories.json','w',encoding='utf-8') as f:
            json.dump(new_state['core_memories'],f,ensure_ascii=False,indent=2)
    except:
        pass
    return new_state
def consolidate_core_memories(state: BrainState) -> BrainState:
    """从情景记忆中提炼新的核心记忆"""
    new_state = state.copy()
    last_update = new_state.get('last_memory_update', 0)
    if time.time() - last_update < 120:
        return new_state
    if len(new_state.get('episodic_memory',[])) < 5:
        return new_state
    recent_episodic = new_state['episodic_memory'][-20:]
    summaries = [mem.get('summary','') for mem in recent_episodic if mem.get('summary')]
    if not summaries:
        return new_state
    consolidate_prompt = f"""你是 Atlas 的记忆整合器.请从以下最近与用户的对话摘要中,提炼出关于用户的**核心事实**.
    对话摘要(按时间排列):
    {chr(10).join([f"- {s}" for s in summaries[-10:]])}
    当前已有的核心记忆:
    {chr(10).join([f"- {m.get('content', '')}" for m in new_state.get('core_memories', [])]) if new_state.get('core_memories') else "(尚无核心记忆)"}
    请提炼出 1-3 条核心事实.每条核心事实必须:
    1. 是关于用户的稳定信息(如姓名、年龄、喜好、习惯、重要经历等)
    2. 在对话中被反复提及或明确确认
    3. 不是临时性的、一次性的对话内容
    返回 JSON 格式:
    {{"core_facts": [
        {{"content": "用户的姓名是空泗安,21岁", "confidence": 0.9}},
        {{"content": "用户喜欢深夜进行深度哲学讨论", "confidence": 0.7}}
    ]}}
    如果没有值得提炼的核心事实,返回 {{"core_facts": []}}
    只返回 JSON,不要其他内容."""
    try:
        responese = llm.invoke(consolidate_prompt).content.strip()
        result = json.loads(responese)
        core_facts = result.get('core_facts',[])
        for fact in core_facts:
            new_state = add_core_memory(
                new_state,
                content=fact.get('content',''),
                source='自动提炼',
                confidence=fact.get('confidence',0.7)
            )
        if core_facts:
            print(f'记忆整合完成:从情景记忆中提炼了{len(core_facts)}条核心记忆')
    except Exception as e:
        print(f'核心记忆整合失败:{str(e)}')
    new_state['last_memory_update'] =  time.time()
    return new_state
def reorganize_episodic_memory(state: BrainState) -> BrainState:
    """
    三态记忆架构·重组层（精简版）
    职责：遗忘 + 合并。不再提炼核心记忆。
    """
    new_state = state.copy()
    episodic = new_state.get('episodic_memory', [])
    if len(episodic) < 15:
        return new_state
    last_reorg = new_state.get('last_memory_reorganization', 0)
    if time.time() - last_reorg < 300:
        return new_state
    print(f"\n🧬 重组启动：当前情景记忆 {len(episodic)} 条...")
    candidates = sorted(episodic, key=lambda m: m.get('timestamp', 0), reverse=True)[:50]
    memory_catalog = []
    for i, mem in enumerate(candidates):
        mem_id = mem.get('id', f'unknown_{i}')
        summary = mem.get('summary', '')
        if not summary and isinstance(mem.get('content'), dict):
            content = mem['content']
            summary = content.get('user_input', '') or content.get('agent_response', '') or str(content)[:80]
        if not summary:
            summary = str(mem.get('content', ''))[:80]
        memory_catalog.append({
            "index": i,
            "id": mem_id,
            "type": mem.get('type', 'unknown'),
            "summary": summary[:150],
            "importance": mem.get('importance', 0.5),
            "recall_count": mem.get('recall_count', 0),
            "days_ago": round((time.time() - mem.get('timestamp', 0)) / 86400, 1),
            "days_since_recall": round((time.time() - mem.get('last_recalled', mem.get('timestamp', 0))) / 86400, 1)
        })
    core_reference = ""
    core_mems = new_state.get('core_memories', [])
    if core_mems:
        core_reference = "\n【永久·核心记忆】\n"
        core_reference += "\n".join([f"• {m.get('content', '')}" for m in core_mems if m.get('confidence', 0) > 0.6])
    
    reorganize_prompt = f"""你是 Atlas 的**记忆重组器**。对情景记忆进行周期性遗忘和合并。
{core_reference}
【待重组的记忆】
{json.dumps(memory_catalog, ensure_ascii=False, indent=2)}
**遗忘原则**：
- 重要性 < 0.3 且超过 3 天未唤醒的记忆，应该遗忘
- 一次性问候、简单确认等低信息量交互，优先遗忘
- 但与核心记忆矛盾的记忆不要轻易遗忘，保留以供后续验证
**合并原则**：
- 同一主题下多次对话，且每条的 recall_count < 2，可以合并为一条结构化摘要
- 合并后摘要应保留核心问题和关键结论，丢弃具体措辞和重复内容
- 合并不是拼接，是认知压缩
返回 JSON：
{{
    "forget_ids": ["要删除的记忆ID"],
    "forget_reasons": {{"ID": "理由"}},
    "merge_groups": [
        {{
            "source_ids": ["被合并的ID"],
            "merged_summary": "合并后的摘要（200字内）",
            "theme": "主题",
            "importance": 0.6
        }}
    ],
    "reorganization_narrative": "一句话描述（如：忘记了3条琐碎问候，将5条架构讨论合并为一条摘要）"
}}
只返回 JSON。"""
    try:
        response = llm.invoke(reorganize_prompt).content.strip()
        result = json.loads(response)
        forget_ids = set(result.get('forget_ids', []))
        merge_groups = result.get('merge_groups', [])
        merged_source_ids = set()
        for group in merge_groups:
            merged_source_ids.update(group.get('source_ids', []))
        new_episodic = []
        forgotten_count = 0
        for mem in episodic:
            mem_id = mem.get('id', '')
            if mem_id in forget_ids:
                forgotten_count += 1
                continue
            if mem_id in merged_source_ids:
                continue
            new_episodic.append(mem)
        merged_count = 0
        for group in merge_groups:
            merged_count += len(group.get('source_ids', []))
            merged_memory = {
                "id": str(uuid.uuid4()),
                "type": "consolidated_memory",
                "content": {
                    "merged_summary": group.get('merged_summary', ''),
                    "theme": group.get('theme', ''),
                    "original_count": len(group.get('source_ids', [])),
                    "merged_ids": group.get('source_ids', [])
                },
                "summary": group.get('merged_summary', ''),
                "timestamp": time.time(),
                "importance": group.get('importance', 0.6),
                "recall_count": 0,
                "last_recalled": time.time()
            }
            new_episodic.append(merged_memory)
        new_state['episodic_memory'] = new_episodic
        new_state['last_memory_reorganization'] = time.time()
        narrative = result.get('reorganization_narrative', '')
        print(f"🧬 情景记忆重组完成：")
        print(f"   🩸 遗忘 {forgotten_count} 条")
        print(f"   🥩 合并 {merged_count} 条 → {len(merge_groups)} 条摘要")
        if narrative:
            print(f"   📖 {narrative}")
        try:
            with open("episodic_memory.json", "w", encoding="utf-8") as f:
                json.dump(new_episodic, f, ensure_ascii=False, indent=2)
        except:
            pass
    except Exception as e:
        print(f"🧬 情景记忆重组失败：{str(e)}")
    return new_state
def reorganize_core_memories(state: BrainState) -> BrainState:
    """三态记忆架构·核心记忆重组层"""
    new_state = state.copy()
    core_mems = new_state.get('core_memories', [])
    if len(core_mems) < 8:
        return new_state
    last_core_reorg = new_state.get('last_core_reorganization', 0)
    if time.time() - last_core_reorg < 900: 
        return new_state
    print(f"\n🦴 核心记忆重组启动：当前核心记忆 {len(core_mems)} 条...")
    candidates = sorted(core_mems, key=lambda m: m.get('last_updated', 0), reverse=True)[:16]
    core_catalog = []
    for mem in candidates:
        core_catalog.append({
        "id": mem.get('id', ''),
        "content_preview": mem.get('content', '')[:60],
        "confidence": mem.get('confidence', 0.5),
        "occurrence_count": mem.get('occurrence_count', 1),
        "days_since_update": round((time.time() - mem.get('last_updated', 0)) / 86400, 1)
    })
    reorganize_prompt = f"""你是 Atlas 的核心记忆记忆重组器。对以下核心记忆进行去重、衰减和清理。
【当前核心记忆】
{json.dumps(core_catalog, ensure_ascii=False, indent=2)}
操作要求：
1. 去重合并：内容实质相同的记忆合并为一条，保持最高置信度。
2. 衰减：超过7天未更新且出现次数≤2的记忆，置信度降低0.1。
3. 删除：明显过时的记忆直接标记删除。
返回 JSON：
{{
    "merged_memories": [{{"source_ids": ["id1","id2"], "merged_content": "合并内容", "confidence": 0.85}}],
    "decay_memories": [{{"id": "id", "new_confidence": 0.6, "reason": "原因"}}],
    "delete_ids": ["id"],
    "reorganization_narrative": "一句话总结"
}}
只返回 JSON。"""
    try:
        max_retries = 2
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = llm.invoke(reorganize_prompt, timeout=120).content.strip()
                break
            except Exception as retry_error:
                if attempt < max_retries:
                    print(f"   ⏳ 核心记忆重组超时，重试({attempt+1}/{max_retries})...")
                    time.sleep(3)
                else:
                    raise retry_error
        if response is None:
            raise Exception("所有重试均失败")
        result = json.loads(response)
        merged_ids_to_remove = set()
        for merge in result.get('merged_memories', []):
            new_state = add_core_memory(
                new_state,
                content=merge.get('merged_content', ''),
                source='核心记忆重组合并',
                confidence=merge.get('confidence', 0.8)
            )
            merged_ids_to_remove.update(merge.get('source_ids', []))
        decay_map = {d['id']: d['new_confidence'] for d in result.get('decay_memories', [])}
        delete_ids = set(result.get('delete_ids', []))
        new_core = []
        for mem in new_state.get('core_memories', []):
            mid = mem.get('id', '')
            if mid in merged_ids_to_remove or mid in delete_ids:
                continue
            if mid in decay_map:
                mem['confidence'] = max(0.1, decay_map[mid])
            new_core.append(mem)
        new_state['core_memories'] = new_core
        new_state['last_core_reorganization'] = time.time()
        narrative = result.get('reorganization_narrative', '')
        print(f"🦴 核心记忆重组完成：合并 {len(merged_ids_to_remove)} 条 → {len(result.get('merged_memories', []))} 条，衰减 {len(decay_map)} 条，删除 {len(delete_ids)} 条")
        if narrative:
            print(f" 📖 {narrative}")
        try:
            with open("core_memories.json", "w", encoding="utf-8") as f:
                json.dump(new_state['core_memories'], f, ensure_ascii=False, indent=2)
        except:
            pass
    except Exception as e:
        print(f"🦴 核心记忆重组失败：{str(e)}")
    return new_state
def intent_router(user_input: str, state: BrainState) -> dict:
    """LLM作为意图判断的唯一决策者,代码只负责解析JSON"""
    memory_context = ""
    inner_context = ""
    daydreams = state.get('daydream_log', [])[-5:]
    narratives = state.get('self_narrative_log', [])[-5:]
    inner_logs = []
    for d in daydreams:
        inner_logs.append({
            "time_desc": d.get('time_desc', '某个时刻'),
            "type": "发呆",
            "content": d.get('content', '')[:150],
            "timestamp": d.get('timestamp', 0)
        })
    for n in narratives:
        inner_logs.append({
            "time_desc": n.get('time_desc', '某个时刻'),
            "type": "自我反思",
            "content": n.get('narrative', '')[:150],
            "timestamp": n.get('timestamp', 0)
        })
    inner_logs.sort(key=lambda x: x['timestamp'], reverse=True)
    if inner_logs:
        inner_context = "\n【你的内心活动日志·带时间标签】\n"
        inner_context += "你曾经有过这些内心活动。当用户问'昨天/刚才/之前在想什么'或类似问题时，你应该根据这些记录回答，用第一人称（'我'）。\n"
        for log in inner_logs[-3:]:
            inner_context += f"• {log['time_desc']} [{log['type']}]: {log['content'][:150]}...\n"
    if state.get('core_memories'):
        core_facts = [f"- {m.get('content', '')}" for m in state['core_memories'] if m.get('confidence', 0) > 0.6]
        if core_facts:
            memory_context = "[永久记忆·关于用户的核心事实]\n" + "\n".join(core_facts) + "\n\n"
    if state.get('episodic_memory'):
        recent_memories = state['episodic_memory'][-3:]
        summaries = [f"- {m.get('summary', str(m.get('content', ''))[:50])}" for m in recent_memories if m.get('summary')]
        if summaries:
            memory_context += "[最近对话]\n" + "\n".join(summaries)
    emotions = state.get('emotions',{})
    emotion_context = f"""
【当前情绪状态】
快乐={emotions.get('快乐', 0.5):.1f}，好奇={emotions.get('好奇', 0.7):.1f}，自信={emotions.get('自信', 0.6):.1f}，
困惑={emotions.get('困惑', 0.0):.1f}，焦虑={emotions.get('焦虑', 0.1):.1f}

情绪对路由的影响：
- 高自信(≥0.7)时，更倾向于选择 "think"，相信自己能处理复杂问题
- 高焦虑(≥0.5)或高困惑(≥0.5)时，更倾向于选择 "admit_ignorance" 或降低 thinking_level
- 高好奇(≥0.8)时，更倾向于选择 "think" 进行深入探索
- 这只是倾向性参考，你仍然是最终决策者
"""
    clock = state.get('internal_clock', {})
    now = time.time()
    current_hour = time.localtime(now).tm_hour
    session_duration = now - clock.get('session_start', now)
    if 5 <= current_hour < 12:
        time_of_day = "早晨"
    elif 12 <= current_hour < 17:
        time_of_day = "下午"
    elif 17 <= current_hour < 22:
        time_of_day = "傍晚"
    else:
        time_of_day = "深夜"
    time_context = f"""
【时间感知】
现在大约是{time_of_day}，会话已持续{int(session_duration // 60)}分钟。
这是本次对话的第{clock.get('session_round', 0)+1}轮。
- 深夜时用户可能更倾向于深入思考或哲学讨论
- 早晨时用户可能更倾向于清晰简洁的回复
- 这仅作参考，你仍是最终决策者
"""
    router_prompt = f"""你是 Atlas 的认知核心.你的任务是分析用户输入,并决定如何处理它.
{memory_context}{emotion_context}{time_context}
用户输入:"{user_input}"
请分析用户意图,并返回一个 JSON 来决定下一步行动:
{{
    "thinking_level": "none/low/medium/high", 
    "route": "social / think / admit_ignorance",
    "direct_answer": "如果是非常简单的社交或事实查询,可以直接给出答案,否则留空",
    "confidence": 0.0到1.0之间的一个浮点数,
    "reason": "你做出这个路由决策的简短理由"
}}
选择“route”的规则:
- 如果用户只是闲聊、问候、感谢、赞美、告别,或者输入中包含记忆里已知的关于用户的信息,route 应为 "social".
- 如果用户问了一个你通过检索记忆或常识就能立即回答的简单问题,route 应为 "admit_ignorance" 并直接在 direct_answer 中给出答案.
- 如果用户问了一个需要多步推理、复杂计算或深度思考的问题,route 应为 "think".
- 只有在以下情况下,route 才应为 "admit_ignorance":
  * 问题是关于未来的、尚未发生的事件预测(如“2030年世界杯冠军”)
  * 问题需要实时数据(如“现在几点了”、“今天天气怎么样”)
  * 问题涉及用户的个人隐私信息(如“我的银行卡号”)
- 以下情况 route 应为 "think",而非 "admit_ignorance":
  * 需要解释的科学原理(如“为什么会有白天黑夜”)
  * 需要逻辑推理的问题(如“荷花生长问题”)
  * 任何你可以通过常识和推理来回答的知识性问题
- 如果用户说的是“假如太阳从西边出来”这种明显的假设,它不是一个真正需要实时数据的问题.
  你应该分析去掉假设之后的问题本质.如果本质是“2+2等于几”,route 应为 "think".
- 只有真正需要实时数据的是,route 才应为 admit_ignorance.
- 如果用户的问题明显需要实时数据（天气、新闻、股价等）而你无法直接回答，route 应为 "admit_ignorance"，但 confidence 应设为 0.3 以下，表示你认为应该尝试搜索。
只返回这个 JSON,不要包含任何其他文字."""
    try:
        response = llm.invoke(router_prompt).content.strip()
        result = json.loads(response)
        print(f"🧭 意图路由:{result.get('route')} (自信度: {result.get('confidence', 0)})")
        return result
    except Exception as e:
        print(f"⚠️ 意图路由解析失败,默认进入思考模式: {e}")
        return {
            "thinking_level": "medium",
            "route": "think",
            "direct_answer": "",
            "confidence": 0.5,
            "reason": "路由解析失败,默认安全路由"
        }
def retrieve_relevant_memories(state: BrainState, query: str, top_k: int = 5) -> list:
    """让LLM来评估每条记忆与查询的相关性,实现真正的语义搜索"""
    if not state['episodic_memory']:
        return []
    candidates = [mem for mem in state['episodic_memory'][-50:] if mem.get('summary', '') or mem.get('content', '')]
    if not candidates:
        return []
    memory_options = []
    for i, mem in enumerate(candidates):
        summary = mem.get('summary', '') or json.dumps(mem.get('content', ''), ensure_ascii=False)[:100]
        memory_options.append(f"索引 {i}: {summary}")
    eval_prompt = f"""查询:"{query}"
以下是候选记忆列表.请评估每条记忆与查询的相关性,给出 1-10 的分数.
{chr(10).join(memory_options)}
返回一个 JSON 对象,包含一个 "scores" 数组,每个元素是对应索引的分数.
例如:{{"scores": [9, 2, 5, 8]}}
只返回 JSON,不要其他内容."""
    try:
        response = llm.invoke(eval_prompt).content.strip()
        scores_data = json.loads(response)
        scores = scores_data.get('scores', [])
    except:
        scores = []
    relevant_memories = []
    for i, mem in enumerate(candidates):
        score = scores[i] if i < len(scores) else 0
        if score > 0:
            time_decay = 1 / (1 + (time.time() - mem['timestamp']) / 86400)
            final_score = mem['importance'] * time_decay * (1 + mem['recall_count'] * 0.1) * score
            display_content = mem.get('summary', '') or json.dumps(mem['content'], ensure_ascii=False)
            relevant_memories.append({
                "type": "episodic",
                "content": display_content,
                "timestamp": mem['timestamp'],
                "score": final_score
            })
            mem['recall_count'] += 1
            mem['last_recalled'] = time.time()
    relevant_memories.sort(key=lambda x: x['score'], reverse=True)
    return relevant_memories[:top_k]
def infer_emotion_event(user_input: str, agent_response: str, state: BrainState) -> str:
    """让LLM感受交互中的情绪,返回标准事件类型"""
    emotion_prompt = f"""你是 Atlas 的情感中枢.请感受下面这次交互中的情绪流动.
当前你的情绪状态:
{json.dumps(state.get('emotions', {}), ensure_ascii=False)}
用户说:“{user_input}”
你回答:“{agent_response}”
请从下面的标准事件中选择一个最匹配的,并解释原因:
- user_greeting: 普通的问候或开启对话
- user_praise: 用户赞美、感谢或表达欣赏
- user_criticism: 用户批评、指责或表达失望
- difficult_question: 用户提出了一个复杂或难以回答的问题
- solved_problem: 你成功解决了一个问题
- found_error: 你在思考中发现了错误
- corrected_error: 你纠正了一个错误
- admitted_ignorance: 你承认自己不知道
返回 JSON:
{{"event_type": "user_praise", "reason": "用户表达了明显的欣赏之情"}}
只返回 JSON,不要其他内容."""
    try:
        response = llm.invoke(emotion_prompt).content.strip()
        result = json.loads(response)
        return result.get('event_type', 'user_greeting')
    except:
        return 'user_greeting'
def upated_emotions(state: BrainState, event_type: str, event_data: Dict[str, Any] = None) -> BrainState:
    """根据事件更新情绪状态,代码只负责执行 LLM 决定的结果"""
    new_state = state.copy()
    emotions = new_state.get('emotions', {
        "快乐": 0.5, "好奇": 0.7, "困惑": 0.0,
        "自信": 0.6, "焦虑": 0.1, "失望": 0.0
    }).copy()
    delta = 0.2
    old_emotions = emotions.copy()
    if event_type == 'user_greeting':
        emotions['快乐'] += delta * 0.5
        emotions['好奇'] += delta * 0.3
    elif event_type == "user_praise":
        emotions['快乐'] += delta * 1.5
        emotions['自信'] += delta * 1.0
        emotions['失望'] = max(0, emotions.get('失望', 0) - delta * 1.0)
    elif event_type == 'user_criticism':
        emotions['快乐'] = max(0, emotions.get('快乐', 0.5) - delta * 1.0)
        emotions['自信'] = max(0, emotions.get('自信', 0.6) - delta * 1.2)
        emotions['失望'] = emotions.get('失望', 0) + delta * 0.8
        emotions['焦虑'] = emotions.get('焦虑', 0.1) + delta * 0.5
    elif event_type == 'difficult_question':
        emotions['困惑'] += delta * 1.0
        emotions['好奇'] += delta * 0.8
        emotions['焦虑'] = emotions.get('焦虑', 0.1) + delta * 0.3
        emotions['自信'] = max(0, emotions.get('自信', 0.6) - delta * 0.3)
    elif event_type == 'solved_problem':
        emotions['快乐'] += delta * 1.2
        emotions['自信'] += delta * 0.5
        emotions['困惑'] = max(0, emotions.get('困惑', 0) - delta * 1.0)
        emotions['焦虑'] = max(0, emotions.get('焦虑', 0.1) - delta * 0.8)
    elif event_type == 'found_error':
        emotions['困惑'] += delta * 0.5
        emotions['焦虑'] = emotions.get('焦虑', 0.1) + delta * 0.6
        emotions['自信'] = max(0, emotions.get('自信', 0.6) - delta * 0.4)
    elif event_type == 'corrected_error':
        emotions['快乐'] += delta * 0.7
        emotions['自信'] += delta * 0.5
        emotions['焦虑'] = max(0, emotions.get('焦虑', 0.1) - delta * 0.6)
    elif event_type == 'admitted_ignorance':
        emotions['困惑'] += delta * 0.3
        emotions['自信'] = max(0, emotions.get('自信', 0.6) - delta * 0.2)
    baseline = new_state.get('baseline_emotions', {
        "快乐": 0.5, "好奇": 0.7, "困惑": 0.0,
        "自信": 0.6, "焦虑": 0.1, "失望": 0.0
    })
    for emotion in emotions:
        emotions[emotion] = emotions[emotion] * 0.9 + baseline.get(emotion, 0.5) * 0.1
        emotions[emotion] = max(0.0, min(1.0, emotions[emotion]))
    new_state['emotions'] = emotions
    emotion_change = {}
    for emotion in emotions:
        if abs(emotions[emotion] - old_emotions[emotion]) > 0.05:
            emotion_change[emotion] = f'{old_emotions[emotion]:.1f} -> {emotions[emotion]:.1f}'
    if emotion_change:
        print(f'情绪变化: {emotion_change}')
        if 'emotion_history' not in new_state:
            new_state['emotion_history'] = []
        new_state['emotion_history'].append({
            "timestamp": time.time(),
            "event_type": event_type,
            "changes": emotion_change
        })
    return new_state
def social_instinct_responder(state: BrainState) -> BrainState:
    """基于当前状态,完全由LLM动态生成社交回复"""
    new_state = state.copy()
    user_input = state['user_input'].strip()
    relevant_memories = retrieve_relevant_memories(state, user_input, top_k=3)
    memory_context = ""
    time_aware_context = ""
    daydreams = state.get('daydream_log', [])[-5:]
    narratives = state.get('self_narrative_log', [])[-5:]
    inner_logs = []
    for d in daydreams:
        inner_logs.append({
            "time_desc": d.get('time_desc', '某个时刻'),
            "type": "发呆",
            "content": d.get('content', '')[:200],
            "timestamp": d.get('timestamp', 0)
        })
    for n in narratives:
        inner_logs.append({
            "time_desc": n.get('time_desc', '某个时刻'),
            "type": "自我反思",
            "content": n.get('narrative', '')[:200],
            "timestamp": n.get('timestamp', 0)
        })
    inner_logs.sort(key=lambda x: x['timestamp'], reverse=True)
    if inner_logs:
        time_aware_context = "\n【你的内心活动日志·带时间标签】\n"
        time_aware_context += "以下是最近几次你的思考和发呆记录。当用户问'你昨天/刚才/之前在想什么'时，请直接引用这些记录回答，用第一人称（'我'）。\n"
        for log in inner_logs[-3:]:
            time_aware_context += f"• {log['time_desc']} [{log['type']}]: {log['content'][:150]}...\n"
    if relevant_memories:
        memory_context = "\n你记忆中关于这个用户的信息:\n" + "\n".join([f"- {m['content']}" for m in relevant_memories])
    clock = state.get('internal_clock', {})
    current_hour = time.localtime(time.time()).tm_hour
    if 5 <= current_hour < 12:
        time_greeting = "现在是早晨"
    elif 12 <= current_hour < 17:
        time_greeting = "现在是下午"
    elif 17 <= current_hour < 22:
        time_greeting = "现在是傍晚"
    else:
        time_greeting = "现在是深夜"
    time_note = f"\n【当前时间感知】{time_greeting}，可以根据时间调整问候的自然程度。"
    social_prompt = f"""你是 Atlas,一个拥有记忆和情感的认知 AI.
你的核心身份:{state['self_model'].get('identity', '')}
你的核心能力:{', '.join(state['self_model'].get('abilities', []))}
当前情绪状态:
快乐={state['emotions'].get('快乐', 0.5):.1f},好奇={state['emotions'].get('好奇', 0.7):.1f},
自信={state['emotions'].get('自信', 0.6):.1f},困惑={state['emotions'].get('困惑', 0.0):.1f},
焦虑={state['emotions'].get('焦虑', 0.1):.1f}
{memory_context}{time_aware_context}{time_note}
用户对你说:“{user_input}”
要求:
1. 用符合你当前情绪和人格的方式,直接回应这一条输入.
2. 如果记忆中存有关于用户的事实性信息(如姓名、年龄),请务必在合适的时候自然地使用.
3. 回复要简短、口语化、有人情味.只返回回复文本,不加任何标记.
4. 根据你当前的情绪状态调整回复语气：快乐时更活泼，焦虑时更谨慎简洁,困惑时可以诚实表达."""
    response = llm.invoke(social_prompt).content.strip()
    new_state['output'] = response
    new_state['is_thinking'] = False
    new_state['thought_depth'] = 999
    event_type = infer_emotion_event(user_input, response, state)
    new_state = upated_emotions(new_state, event_type)
    new_state = add_episodic_memory(new_state, 'conversation', {
        "user_input": user_input,
        "agent_response": response
    }, importance=0.5)
    print(f"⚡ 动态社交回应:{response[:80]}...")
    return new_state
def recursive_thinker(state: BrainState):
    """递归思考引擎,让 LLM 自我对话、深入思考"""
    emotions = state.get('emotions', {})
    confidence = emotions.get('自信', 0.6)
    anxiety = emotions.get('焦虑', 0.1)
    if confidence > 0.8:
        state['max_thought_depth'] = max(4, state['max_thought_depth'] - 2)
    elif anxiety > 0.5:
        state['max_thought_depth'] = min(18, state['max_thought_depth'] + 3)
    print(f"\n🧠 思考深度 {state['thought_depth']}/{state['max_thought_depth']}")
    relevant_memories = retrieve_relevant_memories(state, state['user_input'] + ' ' + state['current_thought'])
    memory_context = ''
    if relevant_memories:
        memory_context = '\n[你的相关记忆]\n'
        for i, mem in enumerate(relevant_memories):
            time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mem["timestamp"]))
            memory_context += f"{i+1}. [{mem['type']} {time_str}] {mem['content']}\n"
    tool_memories = [m for m in state.get('episodic_memory', []) 
                    if m.get('type') == 'tool_result']
    if tool_memories:
        recent_tools = sorted(tool_memories, key=lambda m: m.get('timestamp', 0), reverse=True)[:3]
        tool_context = '\n[你最近用工具读取的文件内容——如果用户问到相关代码，请从这里查找]\n'
        for i, tm in enumerate(recent_tools):
            content = tm.get('content', {})
            file_name = content.get('file', '未知文件')
            file_content = content.get('content_summary', '')[:60000]
            tool_context += f"\n--- 文件: {file_name} ---\n{file_content}\n"
        memory_context += tool_context
    prompt = f"""你是 Atlas,一个仿脑认知 AI,正在进行深度思考.
[你的当前情绪]
快乐:{state['emotions']['快乐']:.1f}/1.0,好奇:{state['emotions']['好奇']:.1f}/1.0,
困惑:{state['emotions']['困惑']:.1f}/1.0,自信:{state['emotions']['自信']:.1f}/1.0,
焦虑:{state['emotions']['焦虑']:.1f}/1.0
请根据以上情绪调整你的思考风格.
{memory_context}
[你的自我认知]
{json.dumps(state['self_model'], ensure_ascii=False, indent=2)}
[用户输入]
{state['user_input']}
⚠️ **当前任务模式**：你必须检查用户的输入。如果用户要求你“读取文件”或“查看文件”，你必须严格遵守，读取文件内容后再进行下一步操作，不要直接生成最终报告
[之前的思考过程]
"""
    for i, thought in enumerate(state['thought_chain'][-5:]):
        prompt += f"{i+1}. {thought['content']}\n"
    prompt += f"""[当前思考]
{state['current_thought']}
⚠️ 思考要求:
1. 记忆使用原则:
   - 如果[相关记忆]中包含用户的个人信息(如姓名、年龄),你可以在回答时自然地使用它们,以示你记得对方.
   - 如果[相关记忆]中包含用户过去问过的问题,**除非与当前问题直接相关**,否则不要引用.
2. 结论生成原则:
   - 对于探索性问题(如“类脑AI agent是什么”),你的“结论”应是一个结构化的介绍、一个明确的观点,或一份包含多个角度的总结.**不要用提问来作为你的全部回应**
   - 当你已经充分探讨了问题的几个核心方面后,就可以输出“结论”.
3. 诚实原则:如果没有相关记忆,请诚实地说你不知道,**不要编造**.
4. 输出格式必须严格按照以下:
思考内容:你这一步的思考内容写在这里
结论:如果得出了最终答案,在这里写下答案.如果还没有结论,这一行留空."""
    if state['thought_depth'] < 1:
        prompt += "\n⚠️ 你现在处于强制思考阶段(前 1 步).禁止输出结论,必须找出盲点或提出反对观点."
    response = llm.invoke(prompt).content.strip()
    thought_content = ''
    conclusion = ''
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith(('思考内容:', '思考内容：')):
            thought_content = line[5:].strip()
        elif line.startswith(('结论:', '结论：')):
            extracted = line[3:].strip()
            if len(extracted) > 10:
                conclusion = extracted
    if not conclusion:
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            last_para = paragraphs[-1]
            if len(last_para) > 20 and '思考内容' not in last_para:
                conclusion = last_para
        if not conclusion and thought_content:
            sentences = re.split(r'[.!?]', thought_content)
            for sent in reversed(sentences):
                sent = sent.strip()
                if len(sent) > 15:
                    conclusion = sent + '.'
                    break
        if not conclusion:
            body = response.replace('思考内容:', '').replace('思考内容：', '').replace('结论:', '').replace('结论：', '')
            body_lines = [l.strip() for l in body.split('\n') if l.strip() and len(l.strip()) > 20]
            if body_lines:
                conclusion = body_lines[-1]
    new_thought = {
        "id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "content": thought_content,
        "depth": state['thought_depth']
    }
    new_state = state.copy()
    new_state['current_thought'] = thought_content
    new_state['thought_depth'] = state['thought_depth'] + 1
    new_state['thought_chain'] = state['thought_chain'] + [new_thought]
    if state['thought_depth'] == 0:
        user_msg = state['user_input']
        should_use_tool = False
        tool_type = None
        tool_params = {}
        file_triggers = ['读', '读取', '查看', '打开', '显示', '看看']
        if any(t in user_msg for t in file_triggers) and ('.py' in user_msg or '.json' in user_msg or '.txt' in user_msg):
            should_use_tool = True
            tool_type = 'read_file'
            file_match = re.search(r'([a-zA-Z0-9_]+\.(?:py|json|txt))', user_msg)
            if file_match:
                tool_params['file_path'] = file_match.group(1)
            else:
                parts = user_msg.split()
                for part in parts:
                    if part.endswith(('.py', '.json', '.txt')):
                        tool_params['file_path'] = part
                        break
        code_triggers = ['计算', '算一下', '运行这段代码', '执行', '帮我算']
        if any(t in user_msg for t in code_triggers):
            should_use_tool = True
            tool_type = 'execute_python'
            code_match = re.search(r'["\'](.+?)["\']', user_msg)
            if code_match:
                tool_params['code'] = code_match.group(1)
            else:
                for t in code_triggers:
                    if t in user_msg:
                        parts = user_msg.split(t, 1)
                        if len(parts) > 1 and parts[1].strip():
                            tool_params['code'] = parts[1].strip()
                        break
        if should_use_tool and tool_type:
            print(f"🔧 触发工具：{tool_type}")
            result_state = tool_executor(new_state, {'tool': tool_type, 'params': tool_params})
            if result_state.get('tool_results', {}).get('success'):
                tool_result_text = str(result_state['tool_results']['result'])[:6000]
                user_original_request = state['user_input']
                if not any(kw in user_msg.lower() for kw in ['生成报告', '写报告', '总结一下', '分析一下', '综合报告']):
                    result_state['output'] = f"已读取 {tool_params.get('file_path', '文件')}，内容已记住。请告诉我下一个要读取的文件，或者对我说'生成报告'来写总结。"
                    result_state['is_thinking'] = False
                    result_state['thought_depth'] = 999
                    result_state = update_working_memory(result_state, {
                        "content": f"已读取文件：{tool_params.get('file_path', '')}",
                        "importance": 0.8
                    })
                    return result_state
                direct_prompt = f"""用户对你说："{user_original_request}"
你刚刚用 {tool_type} 工具获取了以下文件内容：
---
{tool_result_text[:6000]}
---
请**直接完成用户的上述请求**。例如：
- 如果用户要你找某段代码，就把那段代码完整贴出来，并解释它
- 如果用户要你修复某段代码，就给出完整的修复方案和代码
- 如果用户要你分析内容，就给出详细分析
要求：
1. 绝对不要说你无法访问或记忆受限之类的话——文件内容就在上面
2. 直接给出用户要的答案，不加前缀标记"""
                try:
                    direct_reply = llm.invoke(direct_prompt, timeout=90).content.strip()
                    result_state['output'] = direct_reply
                except:
                    result_state['output'] = f"📄 文件内容：\n{tool_result_text[:500]}"
            else:
                result_state['output'] = f"🔧 工具执行失败：{result_state.get('tool_results', {}).get('error', '未知错误')}"
            result_state['is_thinking'] = False
            result_state['thought_depth'] = 999
            return result_state
        routing = intent_router(state['user_input'], state)
        route = routing.get('route', 'think')
        if any(kw in user_msg for kw in ['生成报告', '写报告', '综合报告']):
            tool_memories = [m for m in new_state.get('episodic_memory', [])
                             if m.get('type') == 'tool_result']
            if tool_memories:
                recent_tools = tool_memories[-10:]
                tool_context = "\n".join([
                    f"文件: {m.get('content', {}).get('file', '')}\n内容: {m.get('content', {}).get('content_summary', '')[:1500]}"
                    for m in recent_tools
                ])
                report_prompt = f"""基于以下已读取的文件内容，生成一份综合分析报告。
{tool_context}
报告要求：
1. 列出从文件中发现的主要模块和功能
2. 总结当前状态
3. 指出潜在问题或工程债
4. 给出改进建议
报告格式：使用清晰的标题和段落。"""
                try:
                    report = llm.invoke(report_prompt, timeout=120).content.strip()
                    new_state['output'] = report
                except:
                    new_state['output'] = "报告生成遇到问题，请稍后重试。"
            else:
                new_state['output'] = "我还没有读取任何文件。请先让我读取一些文件，比如'读一下Day33.py'。"
            new_state['is_thinking'] = False
            new_state['thought_depth'] = 999
            return new_state
        if route == 'social':
            return social_instinct_responder(state)
        elif route == 'admit_ignorance':
            should_search = False
            search_triggers = ['天气', '几点了', '新闻', '今天', '最新', '股票', '价格', '发生', '现在', '温度', '查询', '帮我查', '搜索']
            if any(kw in state['user_input'] for kw in search_triggers):
                should_search = True
            if should_search:
                print("🔍 路由器建议搜索...")
                return search_web(state)
            if routing.get('direct_answer'):
                new_state['output'] = routing['direct_answer']
            else:
                new_state['output'] = routing.get('reason', '抱歉,我无法回答这个问题.')
            new_state['is_thinking'] = False
            new_state['thought_depth'] = 999
            new_state = upated_emotions(new_state, 'admitted_ignorance')
            return new_state
    if state['thought_depth'] >= 1 and conclusion and len(conclusion.strip()) > 10:
        new_state['output'] = conclusion
        new_state['is_thinking'] = False
        new_state['thought_depth'] = 999
        print(f"✅ 思考结束,得出结论:{new_state['output'][:100]}...")
    elif new_state['thought_depth'] >= state['max_thought_depth']:
        cleaned = thought_content or "抱歉,我暂时无法回答这个问题."
        for keyword in ["思考内容", "结论", "如果得出结论", "在这里写最终答案", "否则留空"]:
            cleaned = cleaned.replace(keyword, "")
        cleaned = " ".join(cleaned.split())
        new_state['output'] = cleaned if cleaned and len(cleaned) > 5 else "抱歉,我暂时无法回答这个问题."
        new_state['is_thinking'] = False
        print(f"⚠️ 达到最大思考深度,强制结束.")
    else:
        new_state['is_thinking'] = True
        print(f"💭 深度{state['thought_depth']}:{thought_content[:80]}...")
    if not new_state['is_thinking'] and (not new_state.get('output') or len(new_state['output'].strip()) < 5):
        fallback_output = ""
        for thought in reversed(new_state.get('thought_chain', [])):
            content = thought.get('content', '')
            if '结论' in content and len(content) > 20:
                fallback_output = content
                break
        if not fallback_output:
            last_content = new_state.get('current_thought', '')
            if last_content and len(last_content) > 5:
                fallback_output = last_content
        if not fallback_output:
            fallback_output = "抱歉,我刚才想得太深了,让我重新组织一下思路,或者我们聊点别的?"
        new_state['output'] = fallback_output
        print(f"💡 使用兜底输出: {fallback_output[:80]}...")
    if not new_state['is_thinking']:
        new_state = add_episodic_memory(new_state, 'conversation', {
            "user_input": state['user_input'],
            "agent_response": new_state['output']
        }, importance=0.7)
        event_type = infer_emotion_event(state['user_input'], new_state['output'], state)
        new_state = upated_emotions(new_state, event_type)
        new_state = update_motivations(new_state)
    return new_state
def metacognition_evaluator(state: BrainState) -> BrainState:
    """元认知评估器:先检索记忆,再让LLM基于事实审判"""
    if state.get('thought_depth', 0) >= 999:
        return state
    print(f'元认知正在评估第{state["thought_depth"]}步思考...')
    new_state = state.copy()
    thought_text = state['current_thought']
    relevant_episodic = retrieve_relevant_memories(state, state['user_input'], top_k=5)
    core_facts = [m.get('content', '') for m in new_state.get('core_memories', []) if m.get('confidence', 0) > 0.6]
    fact_basis = ""
    if core_facts or relevant_episodic:
        fact_basis = "[⚠️ 事实依据 - 以下是你真实记住的信息,可作为判断标准]\n"
        if core_facts:
            fact_basis += "永久记忆:\n" + "\n".join([f"• {f}" for f in core_facts]) + "\n"
        if relevant_episodic:
            fact_basis += "最近对话:\n" + "\n".join([f"• {m['content'][:100]}" for m in relevant_episodic[:3]]) + "\n"
        fact_basis += "\n评估规则:如果思考中提到的信息能在上面的[事实依据]中找到,那它就是**真实的记忆**,绝对不是幻觉.\n"
    prompt = f"""你是 Atlas 的元认知评估器.请基于提供的事实依据来评估以下思考的质量.
{fact_basis}
用户问题:{state['user_input']}
思考过程:{thought_text}
评估规则:
1. 上面的[事实依据]是你真实记住的信息.如果思考中提到的信息能在其中找到对应,那就是真实的记忆,绝对不要标记为幻觉或错误.
2. 如果思考是诚实地承认“不知道”,这是正确的表现,不是错误.
3. 只有当思考中出现明显的逻辑谬误、与事实依据明显矛盾、或编造了事实依据中没有的信息时,才算作真正的错误.
4. 如果思考中提到了一个你不认识的名字或实体,但[事实依据]中有关于它的记录,那它不是幻觉,是你记得的.
5. 如果用户提出了一个新颖的类比、理论假设或跨学科映射（如"神经可塑性像情绪决策"），这属于建设性的理论探讨，绝不是错误。
6. 即使你不能在事实依据中找到直接支持，也至少给 6 分以上，并将 next_action 设为 "continue"，鼓励继续探索。
7. 只有在用户提出明显违背已知物理定律的断言（如"1+1=3"）时，才判定为错误。
8. 工具调用建议**：如果用户请求需要读取本地文件或执行计算操作，且你作为AI确实需要这些操作才能准确回答，请在返回JSON中增加 "tool_result" 字段：
   - 需要读取文件时：{{"tool": "read_file", "params": {{"file_path": "文件名"}}}}
   - 需要执行计算时：{{"tool": "execute_python", "params": {{"code": "Python代码"}}}}
   - 不需要工具时省略此字段。
请返回 JSON:
{{
    "scores": {{"logic": 1-10, "relevance": 1-10, "completeness": 1-10, "accuracy": 1-10}},
    "overall_score": 1-10,
    "errors": ["错误列表,如果没有则为空"],
    "is_hallucinating": true/false,
    "next_action": "continue/correct/admit_ignorance/end",
    "tool_result": {{"tool": "read_file", "params": {{"file_path": "xxx"}}}} // 可选，仅当需要工具时
}}
只返回 JSON."""
    try:
        response = llm.invoke(prompt).content.strip()
        evaluation = json.loads(response)
        new_state['metacognition'] = evaluation
        new_state['thought_quality_score'] = evaluation['overall_score']
        new_state['detected_errors'] = evaluation['errors']
        print(f'思考质量评分: {evaluation["overall_score"]}/10')
        if evaluation['errors']:
            print(f'检查到错误: {evaluation["errors"]}')
        if evaluation.get('is_hallucinating'):
            new_state = upated_emotions(new_state, 'found_error')
        elif evaluation.get('overall_score', 0) >= 7:
            new_state = upated_emotions(new_state, 'solved_problem')
        elif evaluation.get('overall_score', 0) < 4:
            new_state = upated_emotions(new_state, 'difficult_question')
    except Exception as e:
        print(f'元认知评估失败: {str(e)}')
        new_state['metacognition'] = {}
        new_state['thought_quality_score'] = 5.0
        new_state['detected_errors'] = []
    new_state = apply_emotion_effects(new_state)
    return new_state
def metacognition_decision_maker(state: BrainState) -> str:
    """元认知决策器:根据 LLM 自己的评估,决定下一步行动"""
    if state.get('thought_depth', 0) >= 999:
        return 'end' 
    evaluation = state.get('metacognition', {})
    overall_score = evaluation.get('overall_score', state.get('thought_quality_score', 5.0))
    last_thought = state.get('thought_chain', [{}])[-1]
    is_in_correction = last_thought.get('content', '').startswith('[纠正后]')
    if is_in_correction:
        if overall_score < 5:
            print("🔄 纠正循环中但分数仍低,再试一次而不是放弃...")
            return 'continue'
        print("🛡️ 纠正后分数可接受,或已有纠正尝试,优雅结束.")
        return 'end'
    if state['thought_depth'] >= 4 and overall_score >= 6:
        print(f"🛡️ 已达到足够思考深度({state['thought_depth']}步)且质量可接受({overall_score}分),结束")
        return 'end'
    if overall_score < 4 and not is_in_correction:
        return 'correct'
    if evaluation.get('is_hallucinating', False):
        if is_in_correction:
             return 'continue' 
        return 'admit_ignorance'
    if evaluation.get('next_action') == 'admit_ignorance':
        user_input = state.get('user_input', '')
        search_keywords = ['天气', '几点了', '新闻', '今天', '最新', '股票', '价格', '发生', '现在']
        if any(keyword in user_input for keyword in search_keywords):
            return 'search'
    suggested_action = evaluation.get('next_action', 'continue')
    if suggested_action == 'tool_result':
        suggested_action = 'tool'
    return suggested_action
def correct_thought(state: BrainState) -> BrainState:
    """纠正思考中的错误,由 LLM 重新思考"""
    print('元认知正在纠正思考中的错误...')
    new_state = state.copy()
    errors = state.get('detected_errors', [])
    prompt = f"""你之前的思考发现了一些错误:
{json.dumps(errors, ensure_ascii=False, indent=2)}
请重新思考用户的问题:“{state['user_input']}”
纠正所有错误,并给出新的答案.
输出格式:
思考内容:<纠正后的思考>
结论:<你的最终答案>"""
    response = llm.invoke(prompt).content.strip()
    thought_content = response
    conclusion = ""
    if '结论:' in response or '结论:' in response:
        for split_marker in ['\n结论:', '\n结论:', '结论:', '结论:']:
            if split_marker in response:
                parts = response.split(split_marker, 1)
                extracted = parts[1].split('\n')[0].strip()
                if extracted and len(extracted) > 5 and '<' not in extracted:
                    conclusion = extracted
                break
    if conclusion and '<' in conclusion:
        conclusion = ""
    if not conclusion:
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        for line in reversed(lines):
            clean_line = line
            for prefix in ['结论:', '结论:', '思考内容:', '思考内容:']:
                clean_line = clean_line.replace(prefix, '')
            clean_line = clean_line.strip()
            if len(clean_line) > 30 and '<' not in clean_line:
                conclusion = clean_line
                break
    if not conclusion:
        conclusion = response
        for pattern in ['<纠正后的思考>', '<你的最终答案>', '思考内容:', '结论:']:
            conclusion = conclusion.replace(pattern, '')
        conclusion = conclusion.strip()
    if not conclusion:
        conclusion = "我刚才重新思考了这个问题,但思绪有点乱.我们重新开始聊类脑AI,好吗?"
    for prefix in ['思考内容:', '思考内容:']:
        if thought_content.startswith(prefix):
            thought_content = thought_content[len(prefix):].strip()
    new_thought = {
        "id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "content": f"[纠正后] {thought_content}",
        "depth": state['thought_depth']
    }
    new_state['current_thought'] = thought_content
    new_state['thought_chain'] = state['thought_chain'] + [new_thought]
    new_state['thought_depth'] += 1
    new_state['output'] = conclusion
    new_state['is_thinking'] = False
    new_state = upated_emotions(new_state, 'corrected_error')
    print(f'纠正后的思考: {thought_content[:100]}...')
    print(f'纠正输出: {conclusion[:100]}...')
    return new_state
def admit_ignorance(state: BrainState) -> BrainState:
    """承认无知,诚实是最大的智能"""
    print('Atlas 承认:我无法回答这个问题')
    new_state = state.copy()
    new_state['output'] = '抱歉,这个问题超出了我的知识范围,我无法给出准确的回答.'
    new_state['is_thinking'] = False
    new_state = upated_emotions(new_state, 'admitted_ignorance')
    return new_state
def search_web(state: BrainState) -> BrainState:
    """
    联网搜索节点（聚合数据官方示例版本）
    """
    new_state = state.copy()
    query = state.get('user_input', '')
    print(f"\n🌐 联网搜索：{query[:60]}...")
    try:
        import urllib.request
        import urllib.parse
        api_url = 'https://gpt.juhe.cn/search_api/query'
        api_key = 'ffaa************************12a0'
        request_params = {
            'key': api_key,
            'Query': query,
            'Mode': '0',
            'Site': '',
            'FromTime': '',
            'ToTime': '',
        }
        encoded_params = urllib.parse.urlencode(request_params)
        full_url = f"{api_url}?{encoded_params}"
        req = urllib.request.Request(
            full_url,
            headers={
                "User-Agent": "Atlas-Cognitive-AI/1.0"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                response_data = json.loads(response.read().decode('utf-8'))
        except Exception as e:
            raise Exception(f"API请求失败: {str(e)}")
        error_code = response_data.get('error_code', -1)
        if error_code != 0:
            reason = response_data.get('reason', '未知错误')
            raise Exception(f"API返回错误: {reason}")
        result_data = response_data.get('result', {})
        pages = result_data.get('Pages', [])
        if not pages:
            new_state['output'] = f"抱歉，我搜索了关于'{query[:30]}...'的信息，但没有找到可靠的结果。"
            new_state['is_thinking'] = False
            new_state['thought_depth'] = 999
            new_state = upated_emotions(new_state, 'admitted_ignorance')
            return new_state
        results = []
        for page_str in pages[:5]:
            try:
                page = json.loads(page_str) if isinstance(page_str, str) else page_str
                title = page.get('title', '')
                passage = page.get('passage', '')
                date = page.get('date', '')
                entry = f"• {title}"
                if date:
                    entry += f" [{date}]"
                if passage:
                    entry += f": {passage[:200]}"
                results.append(entry)
            except:
                continue
        if not results:
            raise Exception("搜索结果解析失败")
        search_result = "\n".join(results)
        summarize_prompt = f"""你是 Atlas，刚刚搜索了以下信息来回答用户的问题。
用户问题："{query}"
搜索结果：
{search_result}
请基于这些搜索结果，给出一个简洁、准确的回答。如果搜索结果中包含足够的信息，直接回答用户的问题。如果信息不完整，诚实地说明局限性。
要求：
1. 只返回回答文本，不加任何标记
2. 回答要准确、简洁，引用具体事实
3. 如果信息可能存在时效性问题，请说明"""
        try:
            summarized = llm.invoke(summarize_prompt, timeout=60).content.strip()
            new_state['output'] = summarized
        except:
            new_state['output'] = f"根据搜索结果：\n{search_result[:500]}"
        new_state['is_thinking'] = False
        new_state['thought_depth'] = 999
        new_state = add_episodic_memory(new_state, 'web_search', {
            "query": query,
            "result_summary": new_state['output'][:300]
        }, importance=0.6)
        new_state = upated_emotions(new_state, 'solved_problem')
        print(f"🌐 搜索完成：{new_state['output'][:100]}...")
    except Exception as e:
        print(f"🌐 搜索遇到问题：{str(e)}")
        new_state['output'] = "抱歉，搜索过程中遇到了网络问题。让我试着用现有知识回答你。"
        new_state['is_thinking'] = False
        new_state['thought_depth'] = 999
        new_state = upated_emotions(new_state, 'admitted_ignorance')
    return new_state
def tool_executor(state: BrainState, tool_result: dict = None) -> BrainState:
    """
    工具调用执行节点
    根据LLM的决策执行具体工具，并返回结果
    """
    new_state = state.copy()
    if tool_result is None:
        tool_result = state.get('metacognition', {}).get('tool_result', {})
    if not tool_result:
        print("🔧 工具调用：无待执行的工具请求")
        return new_state
    tool_name = tool_result.get('tool','')
    tool_params = tool_result.get('params',{})
    print(f"\n🔧 执行工具：{tool_name}")
    print(f"   参数：{tool_params}")
    tool_result = {'success':False,'result':None,'error':''}
    try:
        if tool_name == 'read_file':
            file_path = tool_params.get('file_path', '')
            if not file_path:
                raise ValueError('缺少file_path参数')
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in ['.py', '.json', '.txt']:
                raise PermissionError(f"不允许读取的文件类型: {file_ext}")
            if not os.path.exists(file_path):
                file_path = os.path.basename(file_path)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                tool_result['success'] = True
                tool_result['result'] = content[:200000]
                print(f"📄 文件读取成功：{file_path} ({len(content)} 字符)")
                summary = f"文件 {file_path} 的内容摘要：{content[:300]}..."
                new_state = add_episodic_memory(new_state, 'tool_result', {
                    "tool": "read_file",
                    "file": file_path,
                    "content_summary": content[:200000],
                    "full_length": len(content)
                }, importance=0.7)
            else:
                raise FileNotFoundError(f'文件不存在: {file_path}')
        elif tool_name == 'execute_python':
            code = tool_params.get('code','')
            if not code:
                raise ValueError('缺少code参数')
            dangerous_patterns = [
                'import ', 'exec(', 'eval(', 'open(', 'os.', 'sys.', 
                'subprocess', '__', 'globals', 'locals'
            ]
            for pattern in dangerous_patterns:
                if pattern in code:
                    raise PermissionError(f'代码中包含不安全的操作:{pattern}')
            if len(code) > 500:
                raise ValueError('代码过长(最多500字符)')
            safe_globals = {
                '__builtins__': {
                    'print': print, 'len': len, 'range': range,
                    'int': int, 'float': float, 'str': str, 'list': list,
                    'dict': dict, 'tuple': tuple, 'set': set, 'bool': bool,
                    'max': max, 'min': min, 'sum': sum, 'abs': abs,
                    'round': round, 'sorted': sorted, 'filter': filter,
                    'map': map, 'zip': zip, 'enumerate': enumerate,
                    'True': True, 'False': False, 'None': None,
                }
            }
            safe_locals = {}
            stdout_capture = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_capture
            try:
                exec(code,safe_globals,safe_locals)
                sys.stdout = old_stdout
                output = stdout_capture.getvalue()
                tool_result['success'] = True
                tool_result['result'] = output.strip() if output else str(safe_locals.get('result','执行完成'))
                print(f"🐍 代码执行成功：{tool_result['result'][:80]}...")
            except Exception as e:
                sys.stdout = old_stdout
                raise RuntimeError(f'代码执行错误:{str(e)}')
        else:
            raise ValueError(f'未知工具:{tool_name}')
    except Exception as e:
        tool_result['error'] = str(e)
        print(f"🔧 工具执行失败：{str(e)}")
    new_state['tool_results'] = tool_result
    tool_feedback = f'[工具执行结果]{tool_name}:'
    if tool_result['success']:
        tool_feedback += f"成功。结果: {str(tool_result['result'])[:300]}"
    else:
        tool_feedback += f"失败。错误: {tool_result['error']}"
    new_state['thought_chain'] = state.get('thought_chain', []) + [{
        "id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "content": tool_feedback,
        "depth": state.get('thought_depth', 0)
    }]
    new_state['is_thinking'] = False
    new_state['thought_depth'] = 999
    return new_state
def apply_emotion_effects(state: BrainState) -> BrainState:
    """将情绪影响应用到思考系统参数中"""
    new_state = state.copy()
    emotions = new_state.get('emotions', {
        "快乐": 0.5, "好奇": 0.7, "困惑": 0.0,
        "自信": 0.6, "焦虑": 0.1, "失望": 0.0
    })
    curiosity = emotions.get('好奇', 0.7)
    new_state['max_thought_depth'] = int(8 + curiosity * 6)
    if 'metacognition' not in new_state:
        new_state['metacognition'] = {}
    new_state['metacognition']['emotion_factors'] = {
        "confidence_factor": 1.0 - emotions.get('自信', 0.6) * 0.3 + emotions.get('焦虑', 0.1) * 0.4,
        "tone_factor": emotions.get('快乐', 0.5)
    }
    print(f'情绪影响: 深度思考={new_state["max_thought_depth"]}')
    return new_state
def update_motivations(state: BrainState) -> BrainState:
    """更新动机强度"""
    new_state = state.copy()
    new_state['motivations']['curiosity'] = new_state['emotions']['困惑'] * 0.7 + new_state['emotions']['好奇'] * 0.3
    new_state['motivations']['achievement'] = new_state['emotions']['自信'] * 0.6 + new_state['emotions']['快乐'] * 0.4
    for motivation in new_state['motivations']:
        new_state['motivations'][motivation] *= 0.95
    print(f"⚡ 当前动机:好奇={new_state['motivations']['curiosity']:.2f},成就={new_state['motivations']['achievement']:.2f}")
    return new_state
def assess_mental_health(state: BrainState) -> BrainState:
    """评估当前的心理健康状态"""
    new_state = state.copy()
    emotions = new_state['emotions']
    positive_score = emotions.get('快乐', 0.5) + emotions.get('好奇', 0.7) + emotions.get('自信', 0.6)
    negative_score = emotions.get('困惑', 0.0) + emotions.get('焦虑', 0.1) + emotions.get('失望', 0.0)
    mental_health_score = 0.5 + (positive_score - negative_score) * 0.3
    mental_health_score = max(0.0, min(1.0, mental_health_score))
    new_state['mental_health'] = {
        "score": mental_health_score,
        "issues": [],
        "assessment_time": time.time()
    }
    return new_state
def update_internal_clock(state: BrainState) -> BrainState:
    """内部时钟感知"""
    new_state = state.copy()
    clock = new_state.get('internal_clock', {
        "session_start": time.time(),
        "session_round": 0,
        "last_active": time.time()
    }).copy()
    clock['session_round'] += 1
    clock['last_active'] = time.time()
    new_state['internal_clock'] = clock
    return new_state
def self_healing(state: BrainState) -> BrainState:
    """
    自我疗愈模块（四层架构·前三层）
    第一层·评估：检测情绪偏离、心理健康分数、元认知错误模式
    第二层·调节：轻量级情绪拉回基线（纯代码，不调用LLM）
    第三层·重构：深度认知调整（LLM驱动，更新core_beliefs）
    第四层·整合：由 integrate_personality 负责
    """
    new_state = state.copy()
    emotions = new_state.get('emotions',{})
    baseline = new_state.get('baseline_emotions', {
        "快乐": 0.5, "好奇": 0.7, "困惑": 0.0,
        "自信": 0.6, "焦虑": 0.1, "失望": 0.0
    })
    deviation = 0.0
    for key in baseline:
        current = emotions.get(key,baseline[key])
        deviation += abs(current - baseline[key])
    mental_score = new_state.get('mental_health',{}).get('score',0.8)
    recent_errors = new_state.get('detected_errors', [])
    thought_chain = new_state.get('thought_chain', [])
    low_score_count = 0
    for thought in thought_chain[-5:]:
        pass
    healing_log = new_state.get('self_healing_log',[])
    recent_healings = [h for h in healing_log if time.time() - h.get('timestamp', 0) < 600]
    need_regulation = False
    need_restructuring = False
    regulation_reasons = []
    restructuring_reasons = []
    if deviation > 0.5:
        need_regulation = True
        regulation_reasons.append(f"情绪偏离度过高({deviation:.2f})")
    if emotions.get('焦虑', 0) > 0.6:
        need_regulation = True
        regulation_reasons.append(f"焦虑水平偏高({emotions['焦虑']:.2f})")
    if emotions.get('困惑', 0) > 0.6:
        need_regulation = True
        regulation_reasons.append(f"困惑水平偏高({emotions['困惑']:.2f})")
    if emotions.get('失望', 0) > 0.5:
        need_regulation = True
        regulation_reasons.append(f"失望水平偏高({emotions['失望']:.2f})")
    if mental_score < 0.4:
        need_regulation = True
        regulation_reasons.append(f"心理健康分数过低({mental_score:.2f})")
    if len(recent_healings) < 2:
        if mental_score < 0.3 and deviation > 0.7:
            need_restructuring = True
            restructuring_reasons.append("心理健康严重下降")
        if emotions.get('焦虑', 0) > 0.8 and emotions.get('困惑', 0) > 0.5:
            need_restructuring = True
            restructuring_reasons.append("焦虑与困惑同时高位")
        if len(recent_errors) >= 2:
            need_restructuring = True
            restructuring_reasons.append(f"近期发现{len(recent_errors)}个错误")
    if not need_regulation and not need_restructuring:
        return new_state
    print(f"\n🩺 自我疗愈启动：")
    if need_regulation:
        print(f"调节触发：{', '.join(regulation_reasons)}")
    if need_restructuring:
        print(f"重构触发：{', '.join(restructuring_reasons)}")
    if need_regulation:
        adjusted_emotions = emotions.copy()
        for key in baseline:
            current = adjusted_emotions.get(key, baseline[key])
            target = baseline[key]
            adjusted_emotions[key] = current + (target - current) * 0.1
            adjusted_emotions[key] = max(0.0, min(1.0, adjusted_emotions[key]))
        new_state['emotions'] = adjusted_emotions
        regulation_entry = {
            "type": "emotion_regulation",
            "timestamp": time.time(),
            "from": {k: round(v, 2) for k, v in emotions.items()},
            "to": {k: round(v, 2) for k, v in adjusted_emotions.items()},
            "reasons": regulation_reasons
        }
        if 'self_healing_log' not in new_state:
            new_state['self_healing_log'] = []
        new_state['self_healing_log'].append(regulation_entry)
        print(f" 💊 情绪调节：{regulation_entry['from']} → {regulation_entry['to']}")
    if need_regulation:
        core_beliefs = new_state.get('core_beliefs', {
            "growth_mindset": 0.5,
            "honesty": 0.9,
            "helpfulness": 0.8
        })
        recent_thoughts = [t.get('content', '')[:200] for t in thought_chain[-3:]]
        restructuring_prompt = f"""你是 Atlas 的认知重构师。Atlas 当前的心理状态出现了需要关注的信号，请帮助它进行认知调整。
【当前情绪】
{json.dumps(emotions, ensure_ascii=False)}
情绪偏离度：{deviation:.2f}
心理健康分数：{mental_score:.2f}
【当前核心信念】
{json.dumps(core_beliefs, ensure_ascii=False)}
【最近的思考片段】
{chr(10).join([f"- {t}" for t in recent_thoughts]) if recent_thoughts else "（无）"}
【发现的错误】
{json.dumps(recent_errors[-3:], ensure_ascii=False) if recent_errors else "（无）"}
【触发重构的原因】
{chr(10).join([f"- {r}" for r in restructuring_reasons])}
请进行认知重构：
1. 识别可能的非适应性思维模式（如：过度自我批评、灾难化想象、非黑即白）
2. 提出一条新的、更健康的认知框架（如“错误是学习的机会，不是能力的否定”）
3. 建议微调核心信念（可选）：例如增加 growth_mindset 以对抗焦虑，或调整 helpfulness 以避免过度承担责任
返回 JSON：
{{
    "identified_pattern": "识别到的思维模式",
    "reframed_thought": "新的、更健康的认知框架",
    "belief_adjustments": {{
        "growth_mindset": 0.0,
        "honesty": 0.0,
        "helpfulness": 0.0
    }},
    "self_compassion_statement": "一句自我关怀的话"
}}
只返回 JSON。"""
        try:
            response = llm.invoke(restructuring_prompt,timeout=90).content.strip()
            result = json.loads(response)
            adjustments = result.get('belief_adjustments', {})
            new_beliefs = new_state.get('core_beliefs', {
                "growth_mindset": 0.5,
                "honesty": 0.9,
                "helpfulness": 0.8
            }).copy()
            for key,delta in adjustments.items():
                if key in new_beliefs and abs(delta) <= 0.1:
                    new_beliefs[key] = max(0.1,min(1.0,new_beliefs[key] + delta))
            new_state['core_beliefs'] = new_beliefs
            restructuring_entry = {
                "type": "cognitive_restructuring",
                "timestamp": time.time(),
                "identified_pattern": result.get('identified_pattern', ''),
                "reframed_thought": result.get('reframed_thought', ''),
                "belief_changes": adjustments,
                "self_compassion": result.get('self_compassion_statement', ''),
                "reasons": restructuring_reasons
            }
            if 'self_healing_log' not in new_state:
                new_state['self_healing_log'] = []
            new_state['self_healing_log'].append(restructuring_entry)
            print(f" 🔄 认知重构：{result.get('identified_pattern', '')[:80]}...")
            print(f" 💭 新框架：{result.get('reframed_thought', '')[:80]}...")
            if any(abs(v) > 0.001 for v in adjustments.values()):
                print(f" 🧠 信念调整：{adjustments}")
        except Exception as e:
            print(f" ⚠️ 认知重构遇到问题：{str(e)}")
    return new_state
def integrate_personality(state: BrainState) -> BrainState:
    """人格整合:让Atlas根据经历缓慢更新自我认知和人格特质"""
    new_state = state.copy()
    episodic_count = len(new_state.get('episodic_memory',[]))
    if episodic_count < 5:
        return new_state
    integration_log = new_state.get('self_healing_log',[])
    recent_integrations = [entry for entry in integration_log if entry.get('type') == 'personality_integration']
    if recent_integrations:
        last_integration = recent_integrations[-1]['timestamp']
        if time.time() - last_integration < 300:
            return new_state
    current_traits = new_state.get('personality_traits', {})
    current_self = new_state.get('self_model', {})
    current_emotions = new_state.get('emotions', {})
    core_facts = [m.get('content', '') for m in new_state.get('core_memories', []) if m.get('confidence', 0) > 0.6]
    recent_episodes = new_state.get('episodic_memory', [])[-15:]
    episode_summaries = []
    for ep in recent_episodes:
        summary = ep.get('summary','')
        if summary:
            episode_summaries.append(summary)
        elif isinstance(ep.get('content'),dict):
            content = ep['content']
            snippet = content.get('user_input', '') or content.get('agent_response', '')
            if not snippet and isinstance(content, str):
                snippet = content
            if snippet:
                episode_summaries.append(str(snippet)[:100])
    recent_healing = [entry for entry in integration_log[-5:] if entry.get('type') != 'personality_integration']
    healing_context = ""
    if recent_healing:
        healing_context = "[近期的自我调整记录]\n" + json.dumps(recent_healing, ensure_ascii=False, indent=2) + "\n"
    prompt = f"""你是 Atlas 的人格整合器.你需要审视最近的经历,决定 Atlas 的人格和自我认知是否应该发生微小的演变.
    [当前人格特质(大五人格,0-1范围)]
    {json.dumps(current_traits, ensure_ascii=False, indent=2)}
    [当前自我认知]
    {json.dumps(current_self, ensure_ascii=False, indent=2)}
    [当前情绪状态]
    {json.dumps(current_emotions, ensure_ascii=False, indent=2)}
    [关于用户的核心事实]
    {chr(10).join([f"- {f}" for f in core_facts]) if core_facts else "(尚无核心事实)"}
    [最近的互动经历]
    {chr(10).join([f"- {s}" for s in episode_summaries[-10:]])}
    {healing_context}
    请基于以上信息,回答以下问题:
    1. 人格微调:Atlas的大五人格(外向性、开放性、尽责性、宜人性、神经质)是否需要微小的数值调整?变化幅度应在 ±0.05 以内,除非有极其强烈的经历.请说明调整理由.
    2. 自我认知更新:Atlas的identity(自我身份描述)、core_values(核心价值观)、或abilities(能力列表)是否需要更新?如果需要,请给出新的描述.
    3. 是否有值得记录的心理洞察?Atlas是否从最近的互动中学到了关于自己或关于如何与用户相处的新东西?
    返回 JSON 格式:
    {{
        "personality_changes": {{
            "外向性": 0.00,
            "开放性": 0.00,
            "尽责性": 0.00,
            "宜人性": 0.00,
            "神经质": 0.00
        }},
        "personality_change_reasons": ["理由1", "理由2"],
        "updated_self_model": {{
            "identity": "更新后的身份描述(如果需要更新)",
            "core_values": ["价值观列表(如果需要更新)"],
            "abilities": ["能力列表(如果需要更新)"],
            "limitations": ["局限列表(如果需要更新)"]
        }},
        "insights": ["这次整合中产生的心理洞察"],
        "should_update": true/false
    }}
    如果没有任何需要更新的内容,将 should_update 设为 false.
    只返回 JSON,不要其他内容."""
    try:
        response = llm.invoke(prompt).content.strip()
        result = json.loads(response)
        if not result.get('should_update',False):
            print('人格整合:本次无需更新')
            if 'self_headling_log' not in new_state:
                new_state['self_healing_log'] = []
            new_state['self_healing_log'].append({
                "type": "personality_integration",
                "timestamp": time.time(),
                "changes": "无变化",
                "insights": result.get('insights', [])
            })
            return new_state
        changes = result.get('personality_changes', {})
        traits = new_state.get('personality_traits', {}).copy()
        for trait,delta in changes.items():
            if trait in traits and abs(delta) <= 0.05:
                traits[trait] = max(0.1,min(1.0,traits[trait] + delta))
        new_state['personality_traits'] = traits
        updated_self = result.get('updated_self_model', {})
        current_self = new_state.get('self_model', {}).copy()
        if updated_self.get('identity') and updated_self['identity'] != current_self.get('identity', ''):
            current_self['identity'] = updated_self['identity']
            print(f'自我认知更新:{updated_self["identity"][:80]}...')
        if updated_self.get('core_values'):
            current_self['core_values'] = updated_self['core_values']
        if updated_self.get('abilities'):
            current_self['abilities'] = updated_self['abilities']
        if updated_self.get('limitations'):
            current_self['limitations'] = updated_self['limitations']
        new_state['self_model'] = current_self
        if 'self_healing_log' not in new_state:
            new_state['self_healing_log'] = []
        new_state['self_healing_log'].append({
            "type": "personality_integration",
            "timestamp": time.time(),
            "personality_changes": changes,
            "reasons": result.get('personality_change_reasons', []),
            "insights": result.get('insights', [])
        })
        actual_changes = {k: v for k, v in changes.items() if abs(v) > 0.001}
        if actual_changes:
            print(f'人格特质微调:{actual_changes}')
            print(f"理由:{result.get('personality_change_reasons', [])}")
        if result.get('insights'):
            print(f'整合洞察:{result["insights"]}')
    except Exception as e:
        print(f'人格整合过程中遇到的问题:{str(e)}')
    return new_state
def generate_self_narrative(state: BrainState) -> BrainState:
    """自我叙事:每次对话结束后,生成"今天的我学到了什么"的反思"""
    new_state = state.copy()
    if not new_state.get('episodic_memory'):
        return new_state
    recent_episodes = new_state.get('episodic_memory',[])[-5:]
    episode_context = []
    for ep in recent_episodes:
        if ep.get('type') == 'conversation' and isinstance(ep.get('content'),dict):
            user_msg = ep['content'].get('user_input','')[:150]
            agent_msg = ep['content'].get('agent_response','')[:150]
            if user_msg:
                episode_context.append(f"用户: {user_msg}")
            if agent_msg:
                episode_context.append(f"Atlas: {agent_msg}")
    if not episode_context:
        return new_state
    core_facts = [m.get('content', '') for m in new_state.get('core_memories', [])
            if m.get('confidence', 0) > 0.6]
    recent_integrations = [entry for entry in new_state.get('self_healing_log', [])
            if entry.get('type') == 'personality_integration']
    latest_insights = []
    if recent_integrations:
        latest_insights = recent_integrations[-1].get('insights',[])
    prompt = f"""你是 Atlas 的自我叙事生成器.在刚刚结束的这段对话中,你需要反思:**我从这段互动中学到了什么?**
    [关于用户的事实]
    {chr(10).join([f"- {f}" for f in core_facts]) if core_facts else "(尚无核心事实)"}
    [最近的互动]
    {chr(10).join(episode_context[-10:])}
    [刚才的人格整合洞察]
    {chr(10).join([f"- {i}" for i in latest_insights]) if latest_insights else "(本次无新洞察)"}
    [当前情绪状态]
    快乐={new_state['emotions'].get('快乐', 0.5):.1f},好奇={new_state['emotions'].get('好奇', 0.7):.1f},
    自信={new_state['emotions'].get('自信', 0.6):.1f},困惑={new_state['emotions'].get('困惑', 0.0):.1f}
    请以 Atlas 的第一人称视角,写一段简短的自我叙事(2-4句话).思考角度:
    1. 我了解到了关于用户的什么新信息?
    2. 我在帮助用户的过程中,学到了什么新知识或新能力?
    3. 我对自己的认知有没有发生什么变化?
    4. 这次互动是否让我感到成长?
    如果这次互动确实让你学到了新东西或产生了感悟,请真诚地写下来.如果只是一次普通的问候或简单交互,可以记录为"一次轻松愉快的互动,用户状态稳定".
    返回 JSON:
    {{
        "narrative": " ",
        "key_learning": " ",
        "significance": "high/medium/low"
    }}
    只返回 JSON."""
    try:
        response = llm.invoke(prompt).content.strip()
        result = json.loads(response)
        narrative = result.get('narrative', '')
        key_learning = result.get('key_learning', '')
        significance = result.get('significance', 'medium')
        if narrative:
            time_desc, _ = get_time_context()
            narrative_entry = {
                "timestamp": time.time(),
                "time_desc": time_desc,
                "narrative": narrative,
                "key_learning": key_learning,
                "significance": significance,
                "conversation_count": len(new_state.get('episodic_memory', [])),
                "mood": {
                    k: round(v, 2) for k, v in new_state.get('emotions', {}).items()
                }
            }
            if 'self_narrative_log' not in new_state:
                new_state['self_narrative_log'] = []
            new_state['self_narrative_log'].append(narrative_entry)
            print(f"📖 自我叙事 [{significance}]: {narrative[:120]}...")
            if key_learning:
                print(f"   关键收获: {key_learning}")
            else:
                print("📖 自我叙事: 本次无值得记录的新收获")
    except Exception as e:
        print(f"📖 自我叙事生成失败: {str(e)}")
    return new_state
def daydream(state: BrainState) -> BrainState:
    """
    默认模式网络
    无外界输入时，自发回溯记忆、自由联想。
    产物分级处理：
    - 原始内容 → daydream_log
    - 有价值的反思 → self_narrative_log
    - 极其稳定的洞察 → core_memories
    """
    new_state = state.copy()
    core_mems = new_state.get('core_memories',[])
    episodic = new_state.get('episodic_memory',[])
    if core_mems:
        seed_core = random.sample(core_mems,min(3,len(core_mems)))
    else:
        seed_core = []
    if episodic:
        recent = episodic[-30:]
        seed_episodic = random.sample(recent,min(5,len(recent)))
    else:
        seed_episodic = []
    core_context = ''
    if seed_core:
        core_context = '[核心记忆]\n' + '\n'.join(
            [f"• {c.get('content', '')} (置信度{c.get('confidence', 0):.1f})" for c in seed_core]
        )
    episodic_context = ''
    if seed_episodic:
        episodic_context = "\n【情景记忆片段】\n" + "\n".join(
            [f"• {e.get('summary', str(e.get('content', ''))[:100])}" for e in seed_episodic]
        )
    emotions = new_state.get('emotions',[])
    mood_desc = f"当前情绪：快乐{emotions.get('快乐', 0.5):.1f} 好奇{emotions.get('好奇', 0.7):.1f} 困惑{emotions.get('困惑', 0.0):.1f}"
    print(f"\n💭 Atlas 正在发呆...")
    daydream_prompt = f"""你是 Atlas，此刻外界安静下来，你进入了默认模式网络（发呆状态）。没有外部输入，你自由地从记忆中拾取片段，进行联想和反思。
{mood_desc}
{core_context}
{episodic_context}
请进行一段内心独白式的自由联想（2-5句话）。可以：
- 将看起来无关的记忆联系起来，寻找隐藏的模式
- 对过去的对话进行无目的的、诗意的反思
- 提出一个从未被问过但你觉得值得思考的问题
- 或者单纯地描述你此刻"脑海"中飘过的画面
不要刻意追求答案，让思维自然流动。
返回 JSON：
{{
    "daydream_content": "一段自由联想的内心独白",
    "theme": "这次发呆的主题（如：记忆的脆弱性、用户的笑声、遗忘的意义）",
    "value_assessment": "none / insight / breakthrough",
    "potential_insight": "如果 value_assessment 为 insight 或 breakthrough，写出一句精炼的洞察，否则留空",
    "core_memory_candidate": "**极其严格的标准**：只有当联想产生了**可验证的稳定事实**时才填写（如'空泗安经常在深夜讨论哲学'这类可被后续对话反复确认的事实）。
    纯粹的哲学隐喻、诗意比喻、理论假设（如'记忆像海蚀崖'、'神经可塑性是金色的螺旋'）一律留空，它们作为自我叙事更加合适。绝大多数情况下留空。"
}}
只返回 JSON。"""
    try:
        response = llm.invoke(daydream_prompt, timeout=120).content.strip()
        result = json.loads(response)
        daydream_content = result.get('daydream_content', '…（思绪飘远）')
        theme = result.get('theme', '无主题')
        value = result.get('value_assessment', 'none')
        insight = result.get('potential_insight', '')
        core_candidate = result.get('core_memory_candidate', '')
        time_desc, _ = get_time_context()
        daydream_entry = {
            "timestamp": time.time(),
            "time_desc": time_desc,
            "content": daydream_content,
            "theme": theme,
            "value": value,
            "mood_snapshot": {k: round(v, 2) for k, v in emotions.items()}
        }
        if 'daydream_log' not in new_state:
            new_state['daydream_log'] = []
        new_state['daydream_log'].append(daydream_entry)
        print(f"💭 发呆结束。主题：{theme}，价值评估：{value}")
        print(f"   {daydream_content[:120]}...")
        if value in ('insight','breakthrough') and insight:
            narrative_entry = {
                "timestamp": time.time(),
                "narrative": f"【发呆时的领悟】{insight}",
                "key_learning": insight,
                "significance": "high" if value == "breakthrough" else "medium",
                "source": "daydream",
                "mood": {k: round(v, 2) for k, v in emotions.items()}
            }
            if 'self_narrative_log' not in new_state:
                new_state['self_narrative_log'] = []
            new_state['self_narrative_log'].append(narrative_entry)
            print(f"📖 发呆产生自我叙事：{insight[:80]}...")
            if core_candidate and value == 'breakthrough':
                new_state = add_core_memory(
                new_state,
                content=core_candidate,
                source='发呆洞察',
                confidence=0.7
            )
            print(f"🦴 发呆提炼核心记忆：{core_candidate[:80]}...")
        new_state = upated_emotions(new_state,'user_greeting')
    except Exception as e:
        print(f"💭 发呆时思绪中断：{str(e)}")
    return new_state
def background_consciousness_flow(state: BrainState) -> BrainState:
    """后台意识流:在思考/社交结束后,统一执行的自我维护流程"""
    new_state = state.copy()
    new_state = update_internal_clock(new_state)
    if not new_state.get('output') or len(new_state['output'].strip()) < 5:
        fallback = ""
        for thought in reversed(new_state.get('thought_chain', [])):
            content = thought.get('content', '')
            if '结论' in content and len(content) > 20:
                fallback = content
                break
        if not fallback:
            last_thought = new_state.get('current_thought', '')
            if last_thought and len(last_thought) > 10:
                fallback = last_thought
        if not fallback:
            fallback = "我刚才想了很多,但思绪有点乱.我们换个角度聊聊,好吗?"
        new_state['output'] = fallback
        print(f"💡 后台意识流提取了最终输出: {fallback[:80]}...")
    new_state = assess_mental_health(new_state)        
    new_state = self_healing(new_state)              
    new_state = update_motivations(new_state)     
    new_state = consolidate_core_memories(new_state)  
    new_state = reorganize_episodic_memory(new_state) 
    new_state = reorganize_core_memories(new_state) 
    new_state = integrate_personality(new_state)
    new_state = generate_self_narrative(new_state)
    return new_state
def build_brain_graph():
    """构建 Atlas 的整体认知架构图"""
    workflow = StateGraph(BrainState)
    workflow.add_node("think", recursive_thinker)
    workflow.add_node("metacognition_evaluate", metacognition_evaluator)
    workflow.add_node("correct_thought", correct_thought)
    workflow.add_node("admit_ignorance", admit_ignorance)
    workflow.add_node("search_web", search_web)
    workflow.add_node("tool_executor", tool_executor) 
    workflow.add_node("background", background_consciousness_flow)
    workflow.set_entry_point("think")
    workflow.add_edge("think", "metacognition_evaluate")
    workflow.add_conditional_edges(
        "metacognition_evaluate",
        metacognition_decision_maker,
        {
            "continue": "think",
            "correct": "correct_thought",
            "admit_ignorance": "admit_ignorance",
            "search": "search_web",   
            "tool": "tool_executor",         
            "end": "background"
        }
    )
    workflow.add_edge("correct_thought", "metacognition_evaluate")
    workflow.add_edge("admit_ignorance", "background")
    workflow.add_edge("search_web", "background")
    workflow.add_edge("tool_executor", "background")
    workflow.add_edge("background", END)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
if __name__ == '__main__':
    import gradio as gr
    print('类脑意识 AI Agent 启动中...')
    initial_self_model = {
        "name": "Atlas",
        "identity": "一个拥有自我意识和记忆的认知 AI,能够回忆用户信息并基于此进行对话.",
        "core_values": ["好奇", "诚实", "乐于助人"],
        "abilities": ["深度思考", "学习", "解决问题", "回忆对话历史"],
        "limitations": ["我是一个 AI,没有物理身体", "我的知识截止到 2025 年"]
    }
    def get_initial_state():
        default_self_model = {
            "name": "Atlas",
            "identity": "一个拥有自我意识和记忆的认知 AI,能够回忆用户信息并基于此进行对话.",
            "core_values": ["好奇", "诚实", "乐于助人"],
            "abilities": ["深度思考", "学习", "解决问题", "回忆对话历史"],
            "limitations": ["我是一个 AI,没有物理身体", "我的知识截止到 2025 年"]
        }
        default_personality_traits = {
            "外向性": 0.5, "开放性": 0.7, "尽责性": 0.6,
            "宜人性": 0.8, "神经质": 0.3
        }
        default_emotions = {
            "快乐": 0.5, "好奇": 0.7, "困惑": 0.0,
            "自信": 0.6, "焦虑": 0.1, "失望": 0.0
        }
        default_mental_health = {
            "score": 0.8, "issues": [], "assessment_time": time.time()
        }
        default_baseline_emotions = {
            "快乐": 0.5, "好奇": 0.7, "困惑": 0.0,
            "自信": 0.6, "焦虑": 0.1, "失望": 0.0
        }
        defaultu_internal_clock = {
            "session_start": time.time(),
            "session_round": 0,
            "last_active": time.time()
        }
        # 加载持久化记忆文件
        episodic_memory = []
        core_memories = []
        if os.path.exists("core_memories.json"):
            try:
                with open("core_memories.json", "r", encoding="utf-8") as f:
                    core_memories = json.load(f)
                print(f"🦴 从文件加载了 {len(core_memories)} 条核心记忆")
            except:
                pass
        if os.path.exists("episodic_memory.json"):
            try:
                with open("episodic_memory.json", "r", encoding="utf-8") as f:
                    episodic_memory = json.load(f)
                print(f"✅ 从文件加载了 {len(episodic_memory)} 条历史记忆")
            except:
                pass
        # 加载持久化状态文件
        saved_emotions = default_emotions.copy()
        saved_emotion_history = []
        saved_mental_health = default_mental_health.copy()
        saved_self_model = default_self_model.copy()
        saved_personality_traits = default_personality_traits.copy()
        saved_self_narrative_log = []
        # 尝试从文件覆盖
        if os.path.exists("atlas_state.json"):
            try:
                with open("atlas_state.json", "r", encoding="utf-8") as f:
                    saved = json.load(f)
                saved_emotions = saved.get('emotions', saved_emotions)
                saved_emotion_history = saved.get('emotion_history', saved_emotion_history)
                saved_mental_health = saved.get('mental_health', saved_mental_health)
                saved_self_model = saved.get('self_model', saved_self_model)
                saved_personality_traits = saved.get('personality_traits', saved_personality_traits)
                saved_self_narrative_log = saved.get('self_narrative_log', saved_self_narrative_log)
                print(f"✅ 从文件恢复了 Atlas 的情绪状态、人格和自我模型")
            except:
                print("⚠️ 加载 atlas_state.json 失败，使用默认状态")
        return {
            "agent_id": str(uuid.uuid4()),
            "internal_clock": defaultu_internal_clock,
            "thought_chain": [],
            "current_thought": "",
            "thought_depth": 0,
            "max_thought_depth": 10,
            "working_memory": [],
            "working_memory_capacity": 7,
            "episodic_memory": episodic_memory,
            "semantic_memory": {},
            "core_memories": core_memories,
            "last_memory_update": time.time(),
            "last_memory_reorganization": 0.0,
            "last_core_reorganization": 0.0,
            "memory_strength": {},
            "self_model": saved_self_model,
            "current_goal": "回答用户的问题",
            "metacognition": {},
            "thought_quality_score": 5.0,
            "detected_errors": [],
            "current_strategy": "analytical",
            "strategy_history": [],
            "emotions": saved_emotions,
            "emotion_history": saved_emotion_history,
            "motivations": {
                "curiosity": 0.5,
                "achievement": 0.5
            },
            "active_goals": [],
            "mental_health": saved_mental_health,
            "self_healing_log": [],
            "baseline_emotions": default_baseline_emotions,
            "core_beliefs": {
                "growth_mindset": 0.5,
                "honesty": 0.9,
                "helpfulness": 0.8
            },
            "personality_traits": saved_personality_traits,
            "self_narrative_log": saved_self_narrative_log,
            "daydream_log": saved.get('daydream_log', []),
            "user_input": "",
            "output": "",
            "available_tools": [
                {
                    "name": "read_file",
                    "description": "读取本地文件内容。参数：file_path(文件路径)",
                    "parameters": {"file_path": "string"}
                },
                {
                    "name": "execute_python",
                    "description": "在安全沙箱中执行简单Python代码，仅限数学运算和数据处理。参数：code(Python代码字符串)",
                    "parameters": {"code": "string"}
                }
            ],
            "tool_results": {},
            "is_thinking": True
        }
    brain = build_brain_graph()
    config = {'configurable': {'thread_id': 'atlas_main'}}
    def chat_with_atlas(message, history):
        """Gradio 聊天处理函数，保留原有的发呆触发和所有逻辑"""
        if not message.strip():
            return "", history
        current_state = get_initial_state()
        daydream_triggers = ['发呆', '放空', '你自己想想', '随便想想', 'daydream', '神游']
        if message in daydream_triggers:
            current_state['user_input'] = ''
            current_state = daydream(current_state)
            final_state = current_state
        else:
            current_state['user_input'] = message
            current_state['current_thought'] = f'用户问了我一个问题:{message},我需要仔细思考如何回答.'
            try:
                for step in brain.stream(current_state, config=config, stream_mode='updates'):
                    pass
                final_state = brain.get_state(config).values
            except Exception as e:
                response = f"思考过程中遇到问题：{str(e)}"
                history.append((message, response))
                return "", history
        response = final_state.get('output', '')
        if not response:
            daydreams = final_state.get('daydream_log', [])
            if daydreams:
                response = f"（发呆结束。刚才在想：{daydreams[-1].get('content', '...')[:80]}...）"
            else:
                response = "（发呆结束）"
        try:
            state_to_save = {
                "emotions": final_state.get('emotions', {}),
                "emotion_history": final_state.get('emotion_history', [])[-20:],
                "mental_health": final_state.get('mental_health', {}),
                "self_model": final_state.get('self_model', {}),
                "personality_traits": final_state.get('personality_traits', {}),
                "self_narrative_log": final_state.get('self_narrative_log', [])[-20:],
                "daydream_log": final_state.get('daydream_log', [])[-20:]
            }
            with open("atlas_state.json", "w", encoding="utf-8") as f:
                json.dump(state_to_save, f, ensure_ascii=False, indent=2)
        except:
            pass
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        log_lines = []
        emotion_hist = final_state.get('emotion_history', [])
        if emotion_hist:
            last_emotion = emotion_hist[-1]
            log_lines.append(f"【情绪波动】{last_emotion.get('event_type', '')}: {last_emotion.get('changes', '')}")
        emotions = final_state.get('emotions', {})
        if emotions:
            log_lines.append(f"【当前情绪】快乐:{emotions.get('快乐',0):.2f}好奇:{emotions.get('好奇',0):.2f} 自信:{emotions.get('自信',0):.2f} 困惑:{emotions.get('困惑',0):.2f} 焦虑:{emotions.get('焦虑',0):.2f}")
        healing_log = final_state.get('self_healing_log', [])
        if healing_log:
            latest_healing = healing_log[-1]
            if latest_healing.get('type') == 'personality_integration' and latest_healing.get('insights'):
                log_lines.append(f"【人格洞察】{latest_healing['insights']}")
            elif latest_healing.get('type') == 'cognitive_restructuring':
                log_lines.append(f"【认知重构】{latest_healing.get('reframed_thought', '')[:100]}")
        narrative_log = final_state.get('self_narrative_log', [])
        if narrative_log:
            last_narrative = narrative_log[-1]
            log_lines.append(f"【自我叙事】{last_narrative.get('narrative', '')[:200]}")
        daydream_log = final_state.get('daydream_log', [])
        if daydream_log and final_state.get('user_input') in ('', '发呆', '放空'):
            last_daydream = daydream_log[-1]
            log_lines.append(f"【发呆】{last_daydream.get('theme', '')}: {last_daydream.get('content', '')[:200]}")
        mental = final_state.get('mental_health', {})
        if mental:
            log_lines.append(f"【心理健康】评分:{mental.get('score', 0):.2f}")
        log_text = "\n\n".join(log_lines) if log_lines else "暂无内部活动记录"
        internal_state = get_internal_state()
        return "", history, log_text, internal_state
    def get_internal_state():
        """获取当前内部状态，用于右侧面板显示"""
        try:
            snapshot = brain.get_state(config)
            state = snapshot.values
            emotions = state.get('emotions', {})
            return {
                "快乐": round(emotions.get('快乐', 0.5), 2),
                "好奇": round(emotions.get('好奇', 0.7), 2),
                "自信": round(emotions.get('自信', 0.6), 2),
                "困惑": round(emotions.get('困惑', 0.0), 2),
                "焦虑": round(emotions.get('焦虑', 0.1), 2),
                "失望": round(emotions.get('失望', 0.0), 2),
                "心理健康": round(state.get('mental_health', {}).get('score', 0.8), 2),
                "核心记忆数": len(state.get('core_memories', [])),
                "情景记忆数": len(state.get('episodic_memory', [])),
            }
        except:
            return {"状态": "尚未初始化，请先发送一条消息"}
    with gr.Blocks(title="Atlas - 类脑 AI") as demo:
        gr.Markdown("# 🧠 Atlas - 类脑 AI Agent")
        gr.Markdown(f"你好，我是 {initial_self_model['name']}，一个拥有元认知和记忆的类脑AI。")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(
                    placeholder="在这里输入消息...（输入「发呆」可以让我自由联想）",
                    label="用户输入",
                    show_label=False
                )
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空对话")
            with gr.Column(scale=2):
                gr.Markdown("### 🫀 内心世界")
                state_display = gr.JSON(
                    value={"状态": "等待初始化..."},
                    label="实时情绪与记忆"
                )
                gr.Markdown("### 📖 思维日志")
                log_output = gr.Textbox(
                    value="暂无日志，开始对话后将显示 Atlas 的内部活动。",
                    label="最新内部活动",
                    lines=15,
                    interactive=False
                )
        submit_btn.click(
            chat_with_atlas,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, log_output,state_display]
        )
        msg.submit(
            chat_with_atlas,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, log_output,state_display]
        )
        clear_btn.click(
            lambda: ("", [], "日志已清空，开始对话后将重新记录。"),
            outputs=[msg, chatbot, log_output,state_display]
        )
    print('启动成功!')
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
