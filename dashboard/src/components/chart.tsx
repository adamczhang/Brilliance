// A time chart over one scope's samples. The x axis is always the chosen window ending at the
// latest sample, so a model that started a minute ago draws a minute of line in an hour of axis
// instead of stretching it. A sample without a value is a gap in the line, never a zero.
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import uPlot from "uplot"
import "uplot/dist/uPlot.min.css"
import type { Series } from "@/lib/store"
import { fClock, fin } from "@/lib/format"
import { useTheme } from "@/lib/ui"
import { cn } from "@/lib/utils"

export interface Line { key: string; label: string; color?: number; dash?: boolean }
export interface ChartProps {
  data: Series
  version: number
  lines: Line[]
  range: number
  fmt: (v: number) => string
  height?: number
  stack?: boolean
  max?: number        // a fixed upper bound (1 for a ratio)
  tight?: boolean     // the y range follows the data instead of starting at zero
  integer?: boolean   // counts: whole ticks only
  bytes?: boolean     // sizes: ticks at binary multiples
  join?: boolean      // sparse events: join the values across the gaps between them
  className?: string
}

const cssVar = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()
let probe: CanvasRenderingContext2D | null = null
/** A CSS colour as rgba() with the given alpha, whatever notation the stylesheet used. */
export function withAlpha(color: string, a: number): string {
  if (!probe) { const c = document.createElement("canvas"); c.width = c.height = 1; probe = c.getContext("2d", { willReadFrequently: true }) }
  if (!probe) return color
  probe.clearRect(0, 0, 1, 1)
  probe.fillStyle = color
  probe.fillRect(0, 0, 1, 1)
  const [r, g, b] = probe.getImageData(0, 0, 1, 1).data
  return `rgba(${r},${g},${b},${a})`
}
export const seriesColor = (i: number) => cssVar("--series-" + (((i - 1) % 8) + 1))

const INT_STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
const BYTE_STEPS = Array.from({ length: 50 }, (_, i) => 2 ** i)
function niceMax(v: number): number {
  if (!(v > 0)) return 1
  const p = Math.pow(10, Math.floor(Math.log10(v)))
  for (const m of [1, 1.5, 2, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p
  return 10 * p
}

export function Chart({ data, version, lines, range, fmt, height = 220, stack, max, tight, integer, bytes, join, className }: ChartProps) {
  const host = useRef<HTMLDivElement>(null)
  const tip = useRef<HTMLDivElement>(null)
  const plot = useRef<uPlot | null>(null)
  const raw = useRef<(number | null)[][]>([])
  const theme = useTheme()
  const [empty, setEmpty] = useState(true)
  const shape = lines.map((l) => l.key + ":" + (l.color ?? 0) + (l.dash ? "d" : "")).join(",")
  const fmtRef = useRef(fmt)
  fmtRef.current = fmt
  const linesRef = useRef(lines)
  linesRef.current = lines

  // The window's arrays: time, then one array a line (cumulative when stacked).
  const frame = useMemo(() => {
    const end = data.end
    const lo = data.firstAfter(end - range)
    const t = data.t.slice(lo)
    const cols = lines.map((l) => (data.col(l.key) ?? []).slice(lo))
    for (const c of cols) while (c.length < t.length) c.push(null)
    let any = false
    for (const c of cols) for (const v of c) if (v !== null) { any = true; break }
    let drawn: (number | null)[][] = cols
    if (stack) {
      const acc = new Array<number>(t.length).fill(0)
      drawn = cols.map((c) => c.map((v, i) => {
        const some = cols.some((q) => q[i] !== null)
        if (!some) return null
        acc[i] += v ?? 0
        return acc[i]
      }))
    }
    return { t, cols, drawn, end, any }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, version, range, shape, stack])

  useLayoutEffect(() => {
    const el = host.current
    if (!el) return
    const ink = cssVar("--muted-foreground"), grid = cssVar("--grid")
    const font = "11px 'Geist Variable', ui-sans-serif, system-ui, sans-serif"
    const colors = linesRef.current.map((l, i) => (l.dash ? cssVar("--idle") : seriesColor(l.color ?? i + 1)))
    const n = linesRef.current.length
    const opts: uPlot.Options = {
      width: Math.max(120, el.clientWidth), height,
      padding: [10, 22, 0, 0],
      legend: { show: false },
      cursor: { y: false, points: { size: 7, width: 2 }, drag: { x: false, y: false } },
      scales: {
        x: { time: true },
        y: {
          range: (_u, lo, hi) => {
            if (fin(max)) return [0, max]
            if (!fin(lo) || !fin(hi)) return [0, 1]
            if (tight && hi > 0) { const pad = Math.max((hi - lo) * 0.15, hi * 0.01); return [Math.max(0, lo - pad), hi + pad] }
            if (bytes) return [0, Math.max(1024, 2 ** Math.ceil(Math.log2(Math.max(1, hi * 1.02))))]
            return [0, integer ? Math.max(1, Math.ceil(hi)) : niceMax(hi * 1.05)]
          },
        },
      },
      axes: [
        { stroke: ink, font, grid: { show: false }, ticks: { stroke: grid, size: 4 }, space: 58, size: 26, gap: 4,
          values: (u, splits) => { const span = (u.scales.x.max ?? 0) - (u.scales.x.min ?? 0); return splits.map((s) => fClock(s, span <= 600)) } },
        { stroke: ink, font, grid: { stroke: grid, width: 1 }, ticks: { show: false }, gap: 6, space: 28,
          incrs: integer ? INT_STEPS : bytes ? BYTE_STEPS : undefined,
          values: (_u, splits) => splits.map((s) => fmtRef.current(s).replace(/\.0+(?=$|[^\d])/, "")),
          size(self, values, axisIdx, cycle) {
            const axis = self.axes[axisIdx] as uPlot.Axis & { _size?: number }
            if (cycle > 1) return axis._size ?? 40
            let w = 10
            const longest = (values ?? []).reduce((a, v) => (v.length > a.length ? v : a), "")
            if (longest) { self.ctx.font = font; w += self.ctx.measureText(longest).width / devicePixelRatio }
            return Math.ceil(Math.max(34, w))
          } },
      ],
      series: [{}, ...linesRef.current.map((l, i): uPlot.Series => ({
        label: l.label, stroke: colors[i], width: l.dash ? 1.25 : 1.75, dash: l.dash ? [5, 4] : undefined,
        spanGaps: !!join,
        fill: l.dash ? undefined : stack ? withAlpha(colors[i], 0.34) : n === 1 ? withAlpha(colors[i], 0.12) : undefined,
        points: { show: !!join, size: 5, fill: colors[i] },
      }))],
      bands: stack ? linesRef.current.slice(1).map((_, i) => ({ series: [i + 2, i + 1] as [number, number], fill: withAlpha(colors[i + 1], 0.34) })) : undefined,
      hooks: {
        setCursor: [(u) => {
          const box = tip.current
          if (!box) return
          const i = u.cursor.idx
          if (i == null || u.cursor.left == null || u.cursor.left < 0) { box.style.opacity = "0"; return }
          // A chart of many lines (one a model) lists the ones with something at the cursor,
          // largest first, and says how many it left out; a chart of a few lists them all.
          const all = linesRef.current.map((l, k) => ({ l, k, v: raw.current[k]?.[i] ?? null }))
          const many = all.length > 4
          const shown = many ? all.filter((x) => x.v !== null && x.v !== 0).sort((p, q) => (q.v as number) - (p.v as number)).slice(0, 8) : all
          const row = (color: string, label: string, text: string) =>
            `<div class="flex items-center gap-2"><i style="background:${color}" class="size-2 rounded-full shrink-0"></i><span class="text-muted-foreground">${label}</span><b class="ml-auto pl-3 font-medium num">${text}</b></div>`
          const rows = shown.map((x) => row(colors[x.k], x.l.label, x.v == null ? "–" : fmtRef.current(x.v)))
          if (many && stack && shown.length > 1)
            rows.push(`<div class="mt-1 flex border-t pt-1"><span class="text-muted-foreground">Total</span><b class="ml-auto pl-3 font-medium num">${fmtRef.current(shown.reduce((a, x) => a + (x.v as number), 0))}</b></div>`)
          if (many && shown.length === 0) rows.push(`<div class="text-muted-foreground">Nothing at this time</div>`)
          box.innerHTML = `<div class="mb-1 text-muted-foreground num">${fClock(u.data[0][i], true)}</div>` + rows.join("")
          const left = u.cursor.left + (u.bbox.left / devicePixelRatio)
          const flip = left > el.clientWidth * 0.62
          box.style.left = flip ? "" : left + 14 + "px"
          box.style.right = flip ? el.clientWidth - left + 14 + "px" : ""
          box.style.opacity = "1"
        }],
      },
    }
    const u = new uPlot(opts, [[], ...linesRef.current.map(() => [])] as unknown as uPlot.AlignedData, el)
    plot.current = u
    const ro = new ResizeObserver(() => { if (el.clientWidth > 0) u.setSize({ width: el.clientWidth, height }) })
    ro.observe(el)
    return () => { ro.disconnect(); u.destroy(); plot.current = null }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shape, stack, max, tight, integer, bytes, join, height, theme])

  useEffect(() => {
    const u = plot.current
    if (!u) return
    raw.current = frame.cols
    u.batch(() => {
      u.setData([frame.t, ...frame.drawn] as unknown as uPlot.AlignedData, false)
      u.setScale("x", { min: frame.end - range, max: frame.end })
    })
    setEmpty(!frame.any)
  }, [frame, range, theme, shape, stack, max, tight, integer, bytes, join, height])

  return (
    <div className={cn("relative", className)} style={{ height }}>
      <div ref={host} className="absolute inset-0" />
      {empty && <div className="pointer-events-none absolute inset-0 grid place-items-center text-xs text-muted-foreground">No data in this time range</div>}
      <div ref={tip} className="pointer-events-none absolute top-2 z-10 min-w-36 rounded-md border bg-popover px-2.5 py-2 text-xs text-popover-foreground opacity-0 shadow-md transition-opacity" />
    </div>
  )
}

/** A line's colour chips with their names, for a chart of more than one line. */
export function Legend({ lines }: { lines: Line[] }) {
  useTheme()
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      {lines.map((l, i) => (
        <span key={l.key} className="inline-flex items-center gap-1.5">
          <i className="size-2 rounded-full" style={{ background: l.dash ? "var(--idle)" : `var(--series-${((((l.color ?? i + 1) - 1) % 8) + 1)})` }} />
          {l.label}
        </span>
      ))}
    </div>
  )
}
