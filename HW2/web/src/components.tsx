import { Activity, BookOpenText, CircleAlert, Wrench, X } from 'lucide-react'
import { lazy, Suspense } from 'react'

import { artifactUrl } from './api'
import type { AgentResponse, ChatMessage } from './types'

const MarkdownAnswer = lazy(() => import('./MarkdownAnswer'))

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user'
  return (
    <article className={isUser ? 'message user' : 'message assistant'}>
      <div className="message-role">{isUser ? '你' : 'IE-Agent'}</div>
      <div className="message-body">
        {isUser ? (
          <p className="user-copy">{message.content}</p>
        ) : (
          <Suspense fallback={<span className="loading-copy">正在显示回答…</span>}>
            <MarkdownAnswer content={message.content} />
          </Suspense>
        )}

        {message.artifacts?.map((artifact) => (
          <figure className="answer-artifact" key={artifact.name}>
            <a href={artifactUrl(artifact.name)} rel="noreferrer" target="_blank">
              <img alt={artifact.title} loading="lazy" src={artifactUrl(artifact.name)} />
            </a>
            <figcaption>
              <span>{artifact.title}</span>
              <code>{artifact.path}</code>
            </figcaption>
          </figure>
        ))}
      </div>
    </article>
  )
}

export function EvidencePanel({
  response,
  open,
  onClose,
}: {
  response: AgentResponse | null
  open: boolean
  onClose: () => void
}) {
  return (
    <aside className={open ? 'panel evidence-panel is-open' : 'panel evidence-panel'}>
      <div className="panel-heading evidence-heading">
        <span>本轮证据</span>
        <button aria-label="关闭本轮证据" className="close-evidence" onClick={onClose} type="button">
          <X aria-hidden="true" />
        </button>
      </div>

      <div className="evidence-content">
        {response && (
          <>
            <section className="evidence-section">
              <h2>审核门与运行状态</h2>
              <p>{response.outcome === 'passed' ? '审核通过' : response.outcome === 'failed' ? '未通过验收' : '未经审核'} · {response.rounds.length} 轮</p>
              <p>停止原因：{response.stop_reason}</p>
              <p>模型调用 {response.metrics.model_calls} 次 · 输入/输出 token {response.metrics.input_tokens}/{response.metrics.output_tokens}</p>
              <code>{response.artifact_dir}</code>
              {response.rounds.map((round) => (
                <details key={round.number}>
                  <summary>第 {round.number} 轮 · {round.verdict?.approved ? '通过' : '驳回或失败'}</summary>
                  <p>{round.verdict?.summary ?? '尚未获得有效审核结论'}</p>
                  {round.verdict?.issues.map((issue, index) => <p key={index}>{issue}</p>)}
                </details>
              ))}
            </section>
            <section className="evidence-section">
              <h2>Agent 消息</h2>
              {response.messages.map((message, index) => (
                <details key={index}>
                  <summary>第 {message.round_number} 轮 · {message.sender} → {message.recipient}</summary>
                  <pre>{message.content}</pre>
                </details>
              ))}
            </section>
            <section className="evidence-section">
              <h2>
                <Activity aria-hidden="true" />
                调用路径
              </h2>
              <ol className="trace-list">
                {response.trace.map((step) => (
                  <li key={step.step_id}>
                    <span className={`trace-dot ${step.status}`} />
                    <div>
                      <div className="trace-title">
                        <strong>
                          {step.step_id}. {step.name}
                        </strong>
                        {step.latency_ms > 0 && <small>{step.latency_ms.toFixed(1)} ms</small>}
                      </div>
                      <p>{step.summary}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </section>

            {response.tool_results.length > 0 && (
              <section className="evidence-section">
                <h2>
                  <Wrench aria-hidden="true" />
                  本地工具
                </h2>
                <div className="tool-list">
                  {response.tool_results.map((tool) => (
                    <div className="tool-result" key={tool.call_id}>
                      <div className="tool-heading">
                        <code>{tool.tool_name}</code>
                        <span className={`tool-state ${tool.status}`}>
                          {tool.status === 'success' ? '成功' : '失败'}
                        </span>
                      </div>
                      {tool.formula && <code className="tool-formula">{tool.formula}</code>}
                      {tool.steps.length > 0 && (
                        <ol className="tool-steps">
                          {tool.steps.map((step, index) => (
                            <li key={`${tool.call_id}-${index}`}>{step}</li>
                          ))}
                        </ol>
                      )}
                      {Object.keys(tool.result).length > 0 && (
                        <details>
                          <summary>查看结构化结果</summary>
                          <pre>{JSON.stringify(tool.result, null, 2)}</pre>
                        </details>
                      )}
                      {tool.artifacts.map((artifact) => (
                        <code className="artifact-path" key={artifact.name}>
                          {artifact.path}
                        </code>
                      ))}
                      {tool.error && <p className="tool-error">{tool.error}</p>}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {response.citations.length > 0 && (
              <section className="evidence-section">
                <h2>
                  <BookOpenText aria-hidden="true" />
                  教材引用
                </h2>
                <ol className="citation-list">
                  {response.citations.map((citation, index) => (
                    <li key={citation.chunk_id}>
                      <span>{index + 1}</span>
                      <div>
                        <strong>{citation.book_title}</strong>
                        <p>
                          {citation.section}
                          {citation.page ? ` · 第 ${citation.page} 页` : ''}
                        </p>
                      </div>
                    </li>
                  ))}
                </ol>
              </section>
            )}

            {response.warnings.length > 0 && (
              <section className="evidence-section warning-section">
                <h2>
                  <CircleAlert aria-hidden="true" />
                  提示
                </h2>
                <ul>
                  {response.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </section>
            )}
          </>
        )}
      </div>
    </aside>
  )
}
