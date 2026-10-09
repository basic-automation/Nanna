export type PaletteAction = {
  id: string
  label: string
  group: string
  keywords?: string[]
  shortcut?: string
  href?: string
  action?: string
}

/** Primary IA vs Admin — order within groups is list order. */
export const NAV_ACTIONS: PaletteAction[] = [
  // Primary
  { id: 'nav-board', label: 'Board', group: 'Primary', keywords: ['cards', 'tasks', 'kanban', 'todo', 'inbox'], href: '/board' },
  { id: 'nav-chat', label: 'Chat', group: 'Primary', keywords: ['home', 'messages', 'conversation'], href: '/', shortcut: '⌘1' },
  { id: 'nav-memory', label: 'Memory', group: 'Primary', keywords: ['memories', 'knowledge', 'cards'], href: '/memory' },
  { id: 'nav-tools', label: 'Tools', group: 'Primary', keywords: ['mcp', 'functions'], href: '/tools' },
  { id: 'nav-channels', label: 'Channels', group: 'Primary', keywords: ['integrations', 'telegram', 'discord'], href: '/channels' },
  { id: 'nav-settings', label: 'Settings', group: 'Primary', keywords: ['preferences', 'config'], href: '/settings', shortcut: '⌘,' },
  // Admin
  { id: 'nav-logs', label: 'Logs', group: 'Admin', keywords: ['diagnostics', 'console', 'output'], href: '/logs' },
  { id: 'nav-workspaces', label: 'Workspaces', group: 'Admin', keywords: ['projects', 'folders'], href: '/workspaces' },
  { id: 'nav-agents', label: 'Agents', group: 'Admin', keywords: ['personas', 'bots'], href: '/agents' },
  { id: 'nav-scheduler', label: 'Scheduler', group: 'Admin', keywords: ['cron', 'jobs', 'schedule'], href: '/scheduler' },
  { id: 'nav-model-stats', label: 'Model Stats', group: 'Admin', keywords: ['usage', 'tokens', 'llm'], href: '/model-stats' },
  { id: 'nav-tool-stats', label: 'Tool Stats', group: 'Admin', keywords: ['usage', 'mcp'], href: '/tool-stats' },
  { id: 'nav-tool-audit', label: 'Tool Audit', group: 'Admin', keywords: ['audit', 'trail', 'history', 'calls', 'refused'], href: '/tool-audit' },
]

export const QUICK_ACTIONS: PaletteAction[] = [
  { id: 'new-chat', label: 'New chat', group: 'Actions', keywords: ['create', 'session', 'conversation'], action: 'new-chat', shortcut: '⌘N' },
  { id: 'toggle-live-logs', label: 'Toggle live logs', group: 'Actions', keywords: ['pause', 'poll', 'stream'], action: 'toggle-live-logs' },
  { id: 'focus-input', label: 'Focus chat input', group: 'Actions', keywords: ['compose', 'message', 'type'], action: 'focus-input' },
  { id: 'stop-generation', label: 'Stop generation', group: 'Actions', keywords: ['cancel', 'abort', 'halt'], action: 'stop-generation', shortcut: '⌘.' },
  { id: 'open-settings-models', label: 'Open model settings', group: 'Actions', keywords: ['llm', 'provider', 'api'], action: 'open-settings-models' },
  { id: 'toggle-compact-mode', label: 'Toggle compact mode', group: 'Actions', keywords: ['density', 'ui', 'spacing'], action: 'toggle-compact-mode' },
  { id: 'toggle-chat-panel', label: 'Toggle chat panel', group: 'Actions', keywords: ['sidebar', 'sessions', 'drawer'], action: 'toggle-chat-panel' },
]

const RESULT_LIMIT = 50

/**
 * Ceiling for a fuzzy (subsequence) hit, held strictly below the weakest literal tier (group = 30).
 * Fuzzy exists to surface actions a substring search misses entirely — never to reorder the
 * results a literal match already found.
 */
const FUZZY_SCORE_MAX = 25

/** A single-character query is a subsequence of almost everything; requiring two keeps it useful. */
const FUZZY_QUERY_LENGTH_MIN = 2

/** Word boundaries: palette labels are words separated by these. */
function isBoundary(text: string, index: number): boolean {
  if (index === 0) return true
  const previous = text[index - 1]
  return previous === ' ' || previous === '-' || previous === '_'
}

/**
 * Score `query` as a subsequence of `text`, or return -1 when it is not one.
 *
 * Rewards the two things that make a fuzzy hit feel intentional rather than accidental:
 * characters that land consecutively, and characters that land at the start of a word — so
 * `mstats` reaches "Model Stats" and `tglogs` reaches "Toggle live logs". A hit with no
 * word-boundary match at all is rejected: `oes` is technically a subsequence of half the list,
 * and that noise is what makes naive fuzzy palettes feel broken.
 *
 * Single pass over `text`, so cost is bounded by the label length.
 */
export function subsequenceScore(text: string, query: string): number {
  if (query.length < FUZZY_QUERY_LENGTH_MIN) return -1

  let queryIndex = 0
  let raw = 0
  let boundaryHits = 0
  let previousMatchIndex = -2

  for (let index = 0; index < text.length && queryIndex < query.length; index += 1) {
    if (text[index] !== query[queryIndex]) continue
    raw += 1
    if (index === previousMatchIndex + 1) raw += 3
    if (isBoundary(text, index)) {
      raw += 5
      boundaryHits += 1
    }
    previousMatchIndex = index
    queryIndex += 1
  }

  if (queryIndex < query.length) return -1
  if (boundaryHits === 0) return -1

  // Normalise against the best attainable raw score for this query length (every character
  // consecutive and on a boundary), so long labels cannot outrank short ones on length alone.
  const rawMax = query.length * 9
  return Math.round((raw / rawMax) * FUZZY_SCORE_MAX)
}

/**
 * Case-insensitive match on label, group, keywords, id, then a fuzzy subsequence pass over the
 * label and keywords. Empty query → all (order preserved).
 */
export function filterActions(actions: PaletteAction[], query: string): PaletteAction[] {
  const q = query.trim().toLowerCase()
  if (!q) return actions.slice(0, RESULT_LIMIT)

  const scored: Array<{ action: PaletteAction; score: number }> = []
  for (const action of actions) {
    const label = action.label.toLowerCase()
    const group = action.group.toLowerCase()
    const id = action.id.toLowerCase()
    const kw = (action.keywords ?? []).map((k) => k.toLowerCase())

    let score = -1
    if (label === q) score = 100
    else if (label.startsWith(q)) score = 80
    else if (label.includes(q)) score = 60
    else if (id === q || id.includes(q)) score = 50
    else if (kw.some((k) => k === q || k.startsWith(q) || k.includes(q))) score = 40
    else if (group.includes(q)) score = 30
    else score = Math.max(subsequenceScore(label, q), ...kw.map((k) => subsequenceScore(k, q)))

    if (score >= 0) scored.push({ action, score })
  }

  // Array.prototype.sort is stable, so equal scores keep their declaration order.
  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, RESULT_LIMIT).map((s) => s.action)
}
