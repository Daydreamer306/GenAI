export type ChatRole = 'user' | 'assistant'

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  artifacts?: ToolArtifact[]
}

export interface ApiChatMessage {
  role: ChatRole
  content: string
}

export interface Citation {
  source_id: string
  book_title: string
  section: string
  page: number | null
  chunk_id: string
  score: number
  backend: 'qwen' | 'tfidf'
}

export interface ToolResult {
  call_id: string
  tool_name: string
  status: 'success' | 'error'
  result: Record<string, unknown>
  formula: string | null
  steps: string[]
  artifacts: ToolArtifact[]
  latency_ms: number
  error: string | null
}

export interface ToolArtifact {
  name: string
  title: string
  mime_type: string
  path: string
}

export interface RouteDecision {
  intent: string
  planner: 'rules' | 'json' | 'tool_calls' | 'direct_json'
  course_tags: string[]
  need_rag: boolean
  tool_calls: unknown[]
  skill_names: string[]
  reason: string
}

export interface ExecutionStep {
  step_id: number
  stage: 'input' | 'planning' | 'skill' | 'rag' | 'tool' | 'model' | 'answer' | 'review' | 'gate' | 'stop'
  name: string
  status: 'success' | 'error' | 'skipped'
  summary: string
  latency_ms: number
  metadata: Record<string, unknown>
}

export interface ModelResult {
  model: string
  status: 'success' | 'error' | 'fallback'
  latency_ms: number
  input_tokens: number
  output_tokens: number
}

export interface AgentResponse {
  answer: string
  citations: Citation[]
  tool_results: ToolResult[]
  route: RouteDecision
  model_result: ModelResult
  trace: ExecutionStep[]
  warnings: string[]
  outcome: 'passed' | 'failed' | 'unreviewed'
  stop_reason: string
  run_id: string
  artifact_dir: string
  rounds: Array<{
    number: number
    solver_result: ModelResult
    verdict: { approved: boolean; summary: string; issues: string[] } | null
  }>
  messages: Array<{
    round_number: number
    sender: string
    recipient: string
    content: string
  }>
  metrics: {
    execution_mode: 'live' | 'simulated'
    model_calls: number
    input_tokens: number
    output_tokens: number
    total_latency_ms: number
  }
}
