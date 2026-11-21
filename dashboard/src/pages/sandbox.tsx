// A quick check that a model works: load it, unload it, and talk to it in a plain chat. Not the way
// to use the server: apps and agents talk to its API (the note at the top says where).
import { useState } from "react"
import { Play, Square } from "lucide-react"
import { Empty, StateBadge } from "@/components/bits"
import { Chat } from "@/components/chat"
import { Button } from "@/components/ui/button"
import { HttpError, api, type FleetModel } from "@/lib/api"
import { hrefOf } from "@/lib/ui"

const LOCAL_ONLY = "Start and Stop work only from a browser on the server's own machine"
// The states in which a model holds (or is taking) room on its cards.
const HOLDS = new Set(["resident", "loading", "restoring", "parked", "waiting", "closing"])
const cardsOf = (m: FleetModel) => Math.max(m.tp ?? 1, 1)

// `models` is every model the page knows of: the fleet's on the server's own machine, the public
// model list's from elsewhere. `local` is false where the fleet's admin routes are refused (another
// machine, no API key): Start and Stop are then off, and the list and the chat still work.
export function Sandbox({ id, model, models, local }: { id: string; model: FleetModel | undefined; models: FleetModel[]; local: boolean }) {
  const refused = !local
  const [busy, setBusy] = useState<"start" | "stop" | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const base = location.protocol + "//" + location.host
  const state = model?.state
  const caps = model?.capabilities
  // A model that has never been loaded has not told the server what it does yet: assume chat.
  const chats = !caps || !!caps.chat

  // Start holds the model (hold=1): until its Stop, a request for another model is refused rather
  // than moving it off its cards. Stop, an unload, ends the hold.
  async function admin(name: string, action: "load" | "unload") {
    await api(`admin/models/${encodeURIComponent(name)}/${action}${action === "load" ? "?wait=0&hold=1" : ""}`, { method: "POST" })
  }
  async function run(what: "start" | "stop") {
    setBusy(what); setNote(null)
    try {
      if (what === "start") {
        // One model at a time: whatever else holds cards leaves first.
        for (const m of models) if (m.id !== id && HOLDS.has(m.state)) await admin(m.id, "unload")
        await admin(id, "load")
      } else {
        await admin(id, "unload")
      }
    } catch (e) {
      setNote(e instanceof HttpError && e.status === 403 ? LOCAL_ONLY : (e instanceof Error ? e.message : String(e)))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-lg border bg-muted/40 px-3 py-2.5 text-sm text-muted-foreground">
        For simple testing and verification only. Normally you use this server through its API: point an app or agent
        at <code className="num text-foreground">{base}/v1</code> (OpenAI-style) or <code className="num text-foreground">{base}</code> (Anthropic-style).
        {models.length > 1 && <div className="mt-1">Speeds and memory here are not the same as running that profile on its own.</div>}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {models.length > 1 && (
          <select aria-label="Model" value={id} onChange={(e) => { location.hash = hrefOf(e.target.value, "sandbox") }}
            className="h-8 max-w-full min-w-0 rounded-lg border border-input bg-transparent px-2 text-sm dark:bg-input/30">
            {models.map((m) => <option key={m.id} value={m.id} className="bg-background">{m.id} · {cardsOf(m)} card{cardsOf(m) > 1 ? "s" : ""}</option>)}
          </select>
        )}
        <Button size="sm" disabled={refused || busy !== null || state === "resident" || state === "loading"} onClick={() => void run("start")}><Play />{busy === "start" ? "Starting…" : "Start"}</Button>
        <Button size="sm" variant="outline" disabled={refused || busy !== null || !state || !HOLDS.has(state)} onClick={() => void run("stop")}><Square />{busy === "stop" ? "Stopping…" : "Stop"}</Button>
        <StateBadge state={state} doing={model?.doing} />
      </div>
      {(refused || note) && <div className="text-sm text-muted-foreground">{refused ? LOCAL_ONLY + "." : note}</div>}
      {model?.held && HOLDS.has(state ?? "") && <div className="text-sm text-muted-foreground">Held until Stop: requests for other models that need its cards are refused meanwhile.</div>}
      {!model ? <Empty>This process serves no model named {id}</Empty>
        : !chats ? <Empty>This model does not chat (it makes embeddings, pictures or other media), so there is nothing to try here</Empty>
        : <Chat model={id} ready={state === "resident"} hint={state === "loading" || state === "restoring" ? "Loading…" : "Start the model to chat with it"} />}
    </div>
  )
}
