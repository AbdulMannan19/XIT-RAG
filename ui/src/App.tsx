import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface Source {
  url: string
  title: string
  section: string | null
  snippet: string
  char_start: number
  char_end: number
  score: number
}

interface ChatResponse {
  answer_text: string
  sources: Source[]
  confidence: 'low' | 'medium' | 'high'
  query_embedding_similarity: number[]
}

interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
}

const SUGGESTION_CHIPS = [
  {
    text: 'Find IRS Forms (1040, W‑2, 1099)',
    // rgb(67,166,246)
    color: 'from-[#43A6F6] to-[#2E8FE3]',
    bgColor: 'bg-[#43A6F6]',
    hoverColor: 'hover:from-[#2E8FE3] hover:to-[#257ACB]',
  },
  {
    text: 'Deduction & credit guidance',
    // rgb(177,102,142)
    color: 'from-[#B1668E] to-[#9B4F77]',
    bgColor: 'bg-[#B1668E]',
    hoverColor: 'hover:from-[#9B4F77] hover:to-[#844263]',
  },
  {
    text: 'Corporate filing help',
    // rgb(240,162,78)
    color: 'from-[#F0A24E] to-[#DC8E3B]',
    bgColor: 'bg-[#F0A24E]',
    hoverColor: 'hover:from-[#DC8E3B] hover:to-[#C97F33]',
  },
  {
    text: 'Estimate quarterly tax payments',
    // rgb(111,168,116)
    color: 'from-[#6FA874] to-[#5B9362]',
    bgColor: 'bg-[#6FA874]',
    hoverColor: 'hover:from-[#5B9362] hover:to-[#4E8255]',
  },
  {
    text: ' What is the Child Tax Credit?',
    // rgb(124,115,184)
    color: 'from-[#7C73B8] to-[#675EA4]',
    bgColor: 'bg-[#7C73B8]',
    hoverColor: 'hover:from-[#675EA4] hover:to-[#5B5497]',
  },
  {
    text: '1099 vs W‑2 rules',
    // rgb(102,166,152)
    color: 'from-[#66A698] to-[#4F907F]',
    bgColor: 'bg-[#66A698]',
    hoverColor: 'hover:from-[#4F907F] hover:to-[#3E7E6E]',
  },
]

export function App() {
  const [query, setQuery] = useState('')
  const [hasSubmitted, setHasSubmitted] = useState(false)
  const [userQuery, setUserQuery] = useState('')
  const [response, setResponse] = useState<ChatResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set())
  const [messages, setMessages] = useState<Message[]>([])

  function safeHost(u: string) {
    try {
      return new URL(u).hostname
    } catch {
      return u
    }
  }

  async function handleSubmit(submittedQuery: string) {
    if (!submittedQuery.trim()) return

    setHasSubmitted(true)
    setUserQuery(submittedQuery)
    setQuery('')
    setResponse(null)
    setError(null)
    setLoading(true)
    setExpandedSources(new Set())
    setMessages((prev) => [
      ...prev,
      { id: Date.now(), role: 'user', content: submittedQuery },
    ])

    try {
      const res = await fetch('/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: submittedQuery, json: true }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const raw = await res.json()
      const normalized: ChatResponse = {
        answer_text:
          (raw?.answer_text ?? raw?.answer ?? raw?.output ?? raw?.text ?? '').toString(),
        sources: Array.isArray(raw?.sources) ? raw.sources : [],
        confidence: (raw?.confidence === 'low' || raw?.confidence === 'high') ? raw.confidence : 'medium',
        query_embedding_similarity: Array.isArray(raw?.query_embedding_similarity)
          ? raw.query_embedding_similarity
          : [],
      }

      if (!normalized.answer_text) {
        normalized.answer_text = 'No answer text returned.'
      }

      setResponse(normalized)
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: 'assistant',
          content: normalized.answer_text,
          sources: normalized.sources,
        },
      ])
    } catch (e: any) {
      setError(e.message || 'An error occurred')
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 2,
          role: 'assistant',
          content: `Error: ${e.message || 'An error occurred'}`,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function handleChipClick(chipText: string) {
    setQuery(chipText)
    handleSubmit(chipText)
  }


  function toggleSource(key: string) {
    const next = new Set(expandedSources)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    setExpandedSources(next)
  }

  return (
    <div className="min-h-screen bg-white flex flex-col">
      {!hasSubmitted ? (
        // Homepage - Before Query
        <div className="flex-1 flex flex-col items-center justify-center px-4 py-16 md:py-20 lg:py-24">
          {/* Main Title */}
          <h1 className="text-3xl md:text-4xl font-medium text-gray-900 mb-10 md:mb-12 text-center tracking-tight">
            What can I help with?
          </h1>

          {/* Input Bar */}
          <div className="w-full max-w-3xl mb-12 md:mb-16">
            <div className="relative bg-white rounded-full shadow-md hover:shadow-lg focus-within:shadow-lg focus-within:ring-2 focus-within:ring-blue-500/10 transition-all duration-300 border border-gray-100">
              <div className="flex items-center px-6 md:px-7 py-3.5 md:py-4">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      handleSubmit(query)
                    }
                  }}
                  placeholder="Ask anything about IRS tax information"
                  className="flex-1 outline-none text-gray-900 placeholder-gray-400 text-base md:text-lg bg-transparent font-normal"
                />
                <button
                  onClick={() => handleSubmit(query)}
                  disabled={!query.trim()}
                  className="ml-3 text-white bg-gradient-to-r from-gray-800 to-gray-900 px-4 py-2 rounded-full font-semibold shadow-md hover:from-gray-900 hover:to-black disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200"
                  aria-label="Send"
                >
                  <svg
                    className="w-5 h-5"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path d="M5 12h12M13 5l7 7-7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              </div>
            </div>
          </div>

          {/* Suggestion Chips */}
          <div className="w-full max-w-5xl grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5 md:gap-4">
            {SUGGESTION_CHIPS.map((chip, index) => (
              <button
                key={index}
                onClick={() => handleChipClick(chip.text)}
                className={`group relative overflow-hidden bg-gradient-to-r ${chip.color} ${chip.hoverColor} text-white px-5 md:px-6 py-3 md:py-3.5 rounded-full font-medium text-sm md:text-sm flex items-center justify-center gap-2 shadow-md hover:shadow-lg active:scale-[0.98] transition-all duration-300 border border-white/20`}
              >
                <svg
                  className="w-4 h-4 md:w-4 md:h-4 flex-shrink-0"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                </svg>
                <span className="text-center">{chip.text}</span>
              </button>
            ))}
          </div>
        </div>
      ) : (
        // Chat Interface - After Query
        <div className="flex-1 flex flex-col max-w-4xl w-full mx-auto px-4 md:px-6 py-8 md:py-10 bg-white">
          <div className="space-y-6 pb-24 md:pb-28">
            {messages.map((m, idx) => (
              m.role === 'user' ? (
                <div className="flex justify-end" key={m.id}>
                  <div className="bg-gradient-to-r from-gray-800 to-gray-900 text-white px-6 py-4 md:px-7 md:py-4.5 rounded-3xl rounded-tr-md max-w-[85%] md:max-w-[75%] shadow-lg border border-gray-700/50">
                    <p className="text-sm md:text-base leading-relaxed font-normal">{m.content}</p>
                  </div>
                </div>
              ) : (
                <div className="flex justify-start" key={m.id}>
                  <div className="bg-white border border-gray-200/80 px-6 py-5 md:px-7 md:py-6 rounded-3xl rounded-tl-md max-w-[85%] md:max-w-[75%] shadow-lg w-full">
                    <div className="space-y-5">
                      <div className="md-content max-w-none text-gray-800">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {m.content}
                        </ReactMarkdown>
                      </div>
                      {m.sources && m.sources.length > 0 && (
                        <div className="space-y-5">
                          <div>
                            <h3 className="font-semibold text-gray-900 text-base mb-3">IRS Sources</h3>
                            <div className="flex flex-wrap gap-2.5">
                              {m.sources.map((source, index) => (
                                <a
                                  key={index}
                                  href={source.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-700 rounded-full text-xs font-medium hover:from-blue-100 hover:to-indigo-100 transition-all duration-200 border border-blue-200/50 hover:border-blue-300 hover:shadow-md group"
                                >
                                  <span className="max-w-[200px] truncate">{source.title}</span>
                                  <svg className="w-3.5 h-3.5 ml-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                  </svg>
                                </a>
                              ))}
                            </div>
                          </div>

                          {/* Referenced IRS Content (expandable) */}
                          <div className="space-y-2.5">
                            <h4 className="text-sm font-semibold text-gray-800">Referenced IRS Content</h4>
                            {m.sources.map((source, index) => {
                              const key = `${m.id}-${index}`
                              const expanded = expandedSources.has(key)
                              return (
                                <div
                                  key={key}
                                  className="border border-gray-200 rounded-xl overflow-hidden bg-gray-50/50 hover:bg-gray-50 transition-colors duration-200"
                                >
                                  <button
                                    onClick={() => toggleSource(key)}
                                    className="w-full px-5 py-3.5 hover:bg-gray-50/80 transition-colors flex items-center justify-between text-left group"
                                  >
                                    <div className="flex-1 min-w-0 pr-3">
                                      <p className="text-sm font-semibold text-gray-900 truncate group-hover:text-gray-950">
                                        {source.title}
                                        {source.section && (
                                          <span className="text-gray-500 font-normal ml-2">— {source.section}</span>
                                        )}
                                      </p>
                                      <p className="text-xs text-gray-500 mt-1.5 font-medium">{safeHost(source.url)}</p>
                                    </div>
                                    <svg
                                      className={`w-5 h-5 text-gray-400 ml-2 flex-shrink-0 transition-transform duration-200 ${expanded ? 'rotate-180 text-gray-600' : 'group-hover:text-gray-500'}`}
                                      fill="none"
                                      stroke="currentColor"
                                      viewBox="0 0 24 24"
                                    >
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                    </svg>
                                  </button>
                                  {expanded && (
                                    <div className="px-5 py-4 bg-white border-t border-gray-200">
                                      <p className="text-sm text-gray-700 mb-3 whitespace-pre-wrap leading-relaxed">
                                        {source.snippet}
                                      </p>
                                      <div className="flex items-center gap-5 text-xs text-gray-500 mb-3 pb-3 border-b border-gray-100">
                                        <span className="font-medium">Characters: {source.char_start}–{source.char_end}</span>
                                        <span className="font-medium">Relevance: {(source.score * 100).toFixed(1)}%</span>
                                      </div>
                                      <a
                                        href={source.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center gap-1.5 text-sm font-medium text-blue-600 hover:text-blue-700 transition-colors group/link"
                                      >
                                        View source on IRS.gov
                                        <svg className="w-4 h-4 group-hover/link:translate-x-0.5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                        </svg>
                                      </a>
                                    </div>
                                  )}
                                </div>
                              )
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200/80 px-6 py-5 md:px-7 md:py-6 rounded-3xl rounded-tl-md max-w-[85%] md:max-w-[75%] shadow-lg">
                  <div className="flex items-center gap-3 text-gray-600">
                    <div className="flex gap-1.5">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                    <span className="text-sm font-medium">IRS Assistant is thinking…</span>
                  </div>
                </div>
              </div>
            )}
            {error && (
              <div className="flex justify-start">
                <div className="text-red-600 bg-red-50 border border-red-200 rounded-lg p-4">
                  <p className="font-semibold mb-1.5">Error</p>
                  <p className="text-sm">{error}</p>
                </div>
              </div>
            )}
          </div>

          {/* Composer (fixed bottom) */}
          <div className="fixed left-0 right-0 bottom-4 z-40 px-4 md:px-6">
            <div className="relative bg-white rounded-2xl shadow-lg border border-gray-200 max-w-3xl mx-auto">
              <div className="flex items-center px-4 md:px-5 py-3.5">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      if (query.trim()) handleSubmit(query.trim())
                    }
                  }}
                  placeholder="Ask another IRS question…"
                  className="flex-1 outline-none text-gray-900 placeholder-gray-400 bg-transparent text-base"
                />
                <button
                  onClick={() => query.trim() && handleSubmit(query.trim())}
                  disabled={!query.trim() || loading}
                  className="ml-3 text-white bg-gradient-to-r from-gray-800 to-gray-900 px-4 py-2 rounded-xl font-semibold shadow-lg hover:from-gray-900 hover:to-black disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200"
                  aria-label="Send"
                >
                  <svg
                    className="w-5 h-5"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path d="M5 12h12M13 5l7 7-7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
