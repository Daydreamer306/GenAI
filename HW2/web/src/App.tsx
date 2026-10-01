import { PanelRightOpen, RotateCcw, Send } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { sendQuestion } from './api'
import { EvidencePanel, MessageBubble } from './components'
import type { AgentResponse, ApiChatMessage, ChatMessage } from './types'

function createId(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [response, setResponse] = useState<AgentResponse | null>(null)
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [evidenceOpen, setEvidenceOpen] = useState(false)
  const sessionId = useMemo(() => createId('web'), [])
  const endRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, busy])

  async function submit(rawQuestion: string) {
    const query = rawQuestion.trim()
    if (!query || busy) return

    const history: ApiChatMessage[] = messages
      .slice(-8)
      .map(({ role, content }) => ({ role, content }))
    const userMessage: ChatMessage = { id: createId('user'), role: 'user', content: query }
    setMessages((current) => [...current, userMessage])
    setQuestion('')
    setBusy(true)
    setError(null)

    try {
      const result = await sendQuestion(sessionId, query, history)
      const artifacts = result.tool_results.flatMap((tool) => tool.artifacts)
      setResponse(result)
      setMessages((current) => [
        ...current,
        {
          id: createId('assistant'),
          role: 'assistant',
          content: result.answer,
          artifacts,
        },
      ])
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : '请求未完成，请稍后重试。'
      setError(message)
      setMessages((current) => [
        ...current,
        { id: createId('assistant'), role: 'assistant', content: `本轮未完成：${message}` },
      ])
    } finally {
      setBusy(false)
      requestAnimationFrame(() => textareaRef.current?.focus())
    }
  }

  function clearConversation() {
    if (busy) return
    setMessages([])
    setResponse(null)
    setError(null)
    setEvidenceOpen(false)
    textareaRef.current?.focus()
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void submit(question)
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <h1 className="brand">IE-Agent</h1>
        <div className="topbar-actions">
          <span className="ready-state">
            <span className={busy ? 'state-dot is-busy' : 'state-dot'} />
            {busy ? '处理中' : '就绪'}
          </span>
          <button
            className="topbar-button"
            disabled={busy || messages.length === 0}
            onClick={clearConversation}
            type="button"
          >
            <RotateCcw aria-hidden="true" />
            <span>清空</span>
          </button>
          <button
            aria-label="查看本轮证据"
            className="topbar-button evidence-toggle"
            onClick={() => setEvidenceOpen(true)}
            type="button"
          >
            <PanelRightOpen aria-hidden="true" />
          </button>
        </div>
      </header>

      <main className="workspace">
        <section className="panel chat-panel">
          <div className="panel-heading">课程问答</div>
          <div className="conversation" aria-live="polite">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            {busy && (
              <div className="message assistant pending-message">
                <span className="typing-dot" />
                <span className="typing-dot" />
                <span className="typing-dot" />
              </div>
            )}
            <div ref={endRef} />
          </div>

          <div className="composer-area">
            {error && <p className="error-message">{error}</p>}
            <form
              className="composer"
              onSubmit={(event) => {
                event.preventDefault()
                void submit(question)
              }}
            >
              <textarea
                aria-label="课程问题"
                disabled={busy}
                maxLength={4000}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="输入课程问题…"
                ref={textareaRef}
                rows={1}
                value={question}
              />
              <button
                aria-label="发送问题"
                className="send-button"
                disabled={busy || !question.trim()}
                type="submit"
              >
                <Send aria-hidden="true" />
              </button>
            </form>
          </div>
        </section>

        <EvidencePanel
          open={evidenceOpen}
          onClose={() => setEvidenceOpen(false)}
          response={response}
        />
        <button
          aria-label="关闭本轮证据"
          className={evidenceOpen ? 'evidence-overlay is-open' : 'evidence-overlay'}
          onClick={() => setEvidenceOpen(false)}
          type="button"
        />
      </main>
    </div>
  )
}

export default App
