// A terminal-style chat with one model over /v1/chat/completions, streamed. The conversation lives
// in this page only: nothing is stored, and Clear or a reload forgets it.
import { useEffect, useRef, useState } from "react"
import { CornerDownLeft, Eraser, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { auth } from "@/lib/api"

interface Turn { role: "user" | "assistant"; content: string; reasoning?: string; note?: string }

/** Reads an SSE body, calling `onData` with each `data:` payload until `[DONE]` or the end. */
async function readSse(body: ReadableStream<Uint8Array>, onData: (d: string) => void) {
  const reader = body.getReader(), dec = new TextDecoder()
  let buf = ""
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let nl: number
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).replace(/\r$/, "")
      buf = buf.slice(nl + 1)
      if (!line.startsWith("data:")) continue
      const d = line.slice(5).trim()
      if (d === "[DONE]") return
      if (d) onData(d)
    }
  }
}

export function Chat({ model, ready, hint }: { model: string; ready: boolean; hint: string }) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [prompt, setPrompt] = useState("")
  const [running, setRunning] = useState(false)
  const abort = useRef<AbortController | null>(null)
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => { end.current?.scrollIntoView({ block: "nearest" }) }, [turns])
  useEffect(() => () => abort.current?.abort(), [])

  async function send() {
    const text = prompt.trim()
    if (!text || running || !ready) return
    const history: Turn[] = [...turns, { role: "user", content: text }]
    setTurns([...history, { role: "assistant", content: "" }])
    // A reply stopped or failed before any text stays on screen but is not sent back.
    const messages = history.filter((t) => t.role === "user" || t.content).map(({ role, content }) => ({ role, content }))
    setPrompt("")
    setRunning(true)
    const ctl = new AbortController()
    abort.current = ctl
    const patch = (f: (t: Turn) => Turn) => setTurns((ts) => [...ts.slice(0, -1), f(ts[ts.length - 1])])
    try {
      const headers: Record<string, string> = { "Content-Type": "application/json" }
      if (auth.key) headers.Authorization = "Bearer " + auth.key
      const r = await fetch(new URL("v1/chat/completions", document.baseURI), {
        method: "POST", headers, signal: ctl.signal, cache: "no-store",
        body: JSON.stringify({ model, stream: true, messages }),
      })
      if (!r.ok || !r.body) {
        let msg = r.statusText || "HTTP " + r.status
        try { const j = await r.json(); msg = j?.error?.message ?? msg } catch { /* not JSON */ }
        patch((t) => ({ ...t, note: "error: " + msg }))
        return
      }
      await readSse(r.body, (d) => {
        let j: { choices?: { delta?: { content?: string; reasoning_content?: string; reasoning?: string } }[]; error?: { message?: string } }
        try { j = JSON.parse(d) } catch { return }
        if (j.error) { patch((t) => ({ ...t, note: "error: " + (j.error?.message ?? "unknown") })); return }
        const delta = j.choices?.[0]?.delta
        const c = delta?.content ?? "", rz = delta?.reasoning_content ?? delta?.reasoning ?? ""
        if (c || rz) patch((t) => ({ ...t, content: t.content + c, reasoning: rz ? (t.reasoning ?? "") + rz : t.reasoning }))
      })
    } catch (e) {
      patch((t) => ({ ...t, note: ctl.signal.aborted ? "stopped" : "error: " + (e instanceof Error ? e.message : String(e)) }))
    } finally {
      abort.current = null
      setRunning(false)
    }
  }

  return (
    <div className="flex flex-col overflow-hidden rounded-xl border bg-card">
      <div className="max-h-[60vh] min-h-48 overflow-y-auto px-4 py-3 font-mono text-[13px] leading-relaxed">
        {turns.length === 0 && <div className="text-muted-foreground">{ready ? "Type a prompt below. The conversation stays in this page only." : hint}</div>}
        {turns.map((t, i) => (
          <div key={i} className="mb-3 last:mb-0">
            {t.role === "user"
              ? <div className="whitespace-pre-wrap"><span className="text-primary select-none">&gt; </span>{t.content}</div>
              : <>
                  {t.reasoning && (
                    <details className="mb-1 text-muted-foreground/80" open={running && i === turns.length - 1 && !t.content}>
                      <summary className="cursor-pointer text-xs select-none">reasoning</summary>
                      <div className="border-l-2 pl-3 whitespace-pre-wrap opacity-80">{t.reasoning}</div>
                    </details>
                  )}
                  <div className="whitespace-pre-wrap">{t.content}{running && i === turns.length - 1 && <span className="animate-pulse">▍</span>}</div>
                  {t.note && <div className={t.note === "stopped" ? "text-xs text-muted-foreground" : "text-xs text-bad"}>[{t.note}]</div>}
                </>}
          </div>
        ))}
        <div ref={end} />
      </div>
      <form className="flex items-center gap-2 border-t px-3 py-2" onSubmit={(e) => { e.preventDefault(); void send() }}>
        <span className="font-mono text-sm text-primary select-none">&gt;</span>
        <input aria-label="Prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} disabled={!ready || running} autoComplete="off"
          placeholder={ready ? "Ask something" : hint} className="h-8 min-w-0 flex-1 bg-transparent font-mono text-sm outline-none placeholder:text-muted-foreground disabled:opacity-60" />
        {running
          ? <Button type="button" size="sm" variant="outline" onClick={() => abort.current?.abort()}><Square />Stop</Button>
          : <Button type="submit" size="sm" disabled={!ready || !prompt.trim()}><CornerDownLeft />Send</Button>}
        <Button type="button" size="sm" variant="ghost" disabled={running || turns.length === 0} onClick={() => setTurns([])}><Eraser />Clear</Button>
      </form>
    </div>
  )
}
