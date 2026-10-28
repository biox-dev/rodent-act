import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';

type Props = {
  /** Markdown source. `$...$` / `$$...$$` are rendered as KaTeX. */
  source: string;
};

/**
 * Renders a markdown document with GFM tables and KaTeX math.
 *
 * Kept as a tiny standalone component so other pages can reuse the same
 * styling (e.g. a future README tab or an inline help panel).
 */
export default function MarkdownDoc({ source }: Props) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
      >
        {source}
      </ReactMarkdown>
    </div>
  );
}
