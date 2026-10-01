import 'katex/dist/katex.min.css'

import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkMath from 'remark-math'

export default function MarkdownAnswer({ content }: { content: string }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
        {content}
      </ReactMarkdown>
    </div>
  )
}
