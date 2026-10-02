/**
 * The board client's pure half (P25 Stage 4): which column a card sits in,
 * the order inside a column, how a member is named, and which board an event
 * belongs to. No I/O, so the rules are unit-tested rather than eyeballed.
 *
 * The daemon owns every rule that decides anything (blocked is derived there,
 * done is a verdict there); this file only arranges what it returns.
 */

/** A card as `task.list` / `task.get` return it (the fields the board reads). */
export interface BoardCard {
  id: number
  parent_id: number | null
  scope: string
  scope_id: string | null
  title: string
  description: string | null
  status: string
  priority: number
  labels: string[]
  due_at: string | null
  deadline_at: string | null
  assignee: string | null
  /** Derived by the daemon from `depends_on`, never stored. */
  blocked?: boolean
  depends_on?: number[]
  sort_order: number
  created_at: string
  updated_at?: string
}

export interface BoardMember {
  id: string
  name: string
  avatar: string | null
  kind: 'human' | 'agent'
  status: string
  /** `workspace` or `human` (a personal agent, decision 13). */
  owner_kind?: string
  /** Free JSON the router reads whole (model_priority, capabilities, notes…). */
  profile?: unknown
}

/** One post on a card's thread. */
export interface CardPost {
  id: number
  author_member_id: string | null
  author: string | null
  kind: string
  content: string
  created_at: string
}

export type ColumnId = 'todo' | 'blocked' | 'in_progress' | 'done'

export interface BoardColumn {
  id: ColumnId
  label: string
  cards: BoardCard[]
}

export const COLUMNS: ReadonlyArray<{ id: ColumnId, label: string }> = [
  { id: 'todo', label: 'To do' },
  { id: 'blocked', label: 'Waiting' },
  { id: 'in_progress', label: 'In progress' },
  { id: 'done', label: 'Done' },
]

/**
 * How many closed cards the Done column shows, newest first.
 *
 * Bound: a board accumulates closed cards forever (threads are permanent,
 * P25 decision 14) and the list read returns them all; a column of hundreds
 * of finished cards buries the open work. Fifty is a few weeks of a busy
 * board and still one screen of scrolling.
 */
export const DONE_SHOWN_MAX = 50

/** The router's member ids start with this (`router:global`, `router:<ws>`). */
export const ROUTER_PREFIX = 'router:'

/**
 * Which column a card belongs in. A blocked card waits (on a clarification
 * or another card) whatever its stored status says, unless it is closed.
 */
export function columnOf(card: BoardCard): ColumnId {
  if (card.status === 'done' || card.status === 'cancelled') return 'done'
  if (card.blocked) return 'blocked'
  if (card.status === 'in_progress') return 'in_progress'
  return 'todo'
}

/** Open columns: priority (p1 first), then the board's own order, then age. */
function compareOpen(a: BoardCard, b: BoardCard): number {
  return a.priority - b.priority
    || a.sort_order - b.sort_order
    || a.id - b.id
}

/** Done: most recently finished first. */
function compareDone(a: BoardCard, b: BoardCard): number {
  return (b.updated_at ?? b.created_at).localeCompare(a.updated_at ?? a.created_at) || b.id - a.id
}

/**
 * Arrange cards into the four columns. `nested` hides sub-cards (the board's
 * flat/nested toggle, P25 decision 8) — a sub-card is still counted on its
 * parent via {@link childCounts}.
 */
export function arrangeColumns(cards: readonly BoardCard[], nested: boolean): BoardColumn[] {
  const shown = nested ? cards.filter(card => card.parent_id === null) : cards
  const columns = COLUMNS.map(column => ({ ...column, cards: [] as BoardCard[] }))
  const byId = new Map(columns.map(column => [column.id, column]))
  for (const card of shown) byId.get(columnOf(card))?.cards.push(card)
  for (const column of columns) {
    column.cards.sort(column.id === 'done' ? compareDone : compareOpen)
    if (column.id === 'done') column.cards = column.cards.slice(0, DONE_SHOWN_MAX)
  }
  return columns
}

/** Open and total sub-card counts per parent id. */
export function childCounts(cards: readonly BoardCard[]): Map<number, { open: number, total: number }> {
  const counts = new Map<number, { open: number, total: number }>()
  for (const card of cards) {
    if (card.parent_id === null) continue
    const entry = counts.get(card.parent_id) ?? { open: 0, total: 0 }
    entry.total += 1
    if (columnOf(card) !== 'done') entry.open += 1
    counts.set(card.parent_id, entry)
  }
  return counts
}

/** The name to show for a member id: the roster's name, else the id itself. */
export function memberName(id: string | null | undefined, roster: readonly BoardMember[]): string {
  if (!id) return 'Unassigned'
  return roster.find(member => member.id === id)?.name ?? id
}

/** Members a card can be assigned to — everyone but the router (decision 4). */
export function assignable(roster: readonly BoardMember[]): BoardMember[] {
  return roster.filter(member => !member.id.startsWith(ROUTER_PREFIX))
}

/** Up to two initials for an avatar disc. */
export function initials(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean)
  const letters = words.length > 1 ? [words[0] ?? '', words[words.length - 1] ?? ''] : words
  return letters.map(word => Array.from(word)[0] ?? '').join('').toUpperCase() || '?'
}

/** `YYYY-MM-DD` of a stored date or timestamp. */
export function dayOf(value: string | null | undefined): string | null {
  return value ? value.slice(0, 10) : null
}

/** A deadline before `today` on an open card. */
export function isOverdue(card: BoardCard, today: string): boolean {
  const deadline = dayOf(card.deadline_at)
  return deadline !== null && deadline < today && columnOf(card) !== 'done'
}

/** A defer date still in the future: the card is not startable yet. */
export function isDeferred(card: BoardCard, today: string): boolean {
  const due = dayOf(card.due_at)
  return due !== null && due > today
}

/** Today as the store's `YYYY-MM-DD` (its days are UTC). */
export function todayUtc(now: Date = new Date()): string {
  return now.toISOString().slice(0, 10)
}

/**
 * Whether a `board-event` belongs to the board being shown: the same scope,
 * and for a workspace board the same workspace.
 */
export function eventIsForBoard(
  event: { scope?: string, scope_id?: string | null },
  board: { scope: 'workspace' | 'global', workspaceId: string | null },
): boolean {
  if (event.scope !== board.scope) return false
  return board.scope === 'global' || (event.scope_id ?? null) === board.workspaceId
}

/** The thread's label for a post kind. */
export function postKindLabel(kind: string): string {
  return ({ comment: 'commented', progress: 'progress', question: 'asked', verdict: 'verdict' } as Record<string, string>)[kind] ?? kind
}

/** One day of the Upcoming list. */
export interface UpcomingDay {
  day: string
  cards: BoardCard[]
}

/**
 * Split a member's open cards (from `list_assigned_cards`) into the Inbox —
 * no date, or a date today or past (P25 decision 11: no date means now) —
 * and Upcoming, grouped by date ascending. `today` is the store's UTC day,
 * which the daemon sends with the cards.
 */
export function splitAssigned(cards: readonly BoardCard[], today: string): { inbox: BoardCard[], upcoming: UpcomingDay[] } {
  const inbox: BoardCard[] = []
  const byDay = new Map<string, BoardCard[]>()
  for (const card of cards) {
    const day = dayOf(card.due_at)
    if (day === null || day <= today) {
      inbox.push(card)
      continue
    }
    const list = byDay.get(day) ?? []
    list.push(card)
    byDay.set(day, list)
  }
  inbox.sort(compareOpen)
  const upcoming = [...byDay.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([day, list]) => ({ day, cards: list.sort(compareOpen) }))
  return { inbox, upcoming }
}

/** The board a card is on, by name: its workspace's, else "Global". */
export function boardLabel(card: BoardCard, workspaces: ReadonlyArray<{ id: string, name: string, path: string }>): string {
  if (card.scope !== 'workspace') return 'Global'
  const ws = workspaces.find(w => w.id === card.scope_id)
  return ws ? (ws.name || ws.path) : (card.scope_id ?? 'Workspace')
}

/** The member-profile fields the board edits; the router reads the whole profile. */
export interface ProfileForm {
  /** Comma- or newline-separated model ids, best first (`model_priority`). */
  models: string
  /** Comma-separated free-text capability tags (decision 15). */
  capabilities: string
  /** Free text the router sees: what this member is for. */
  notes: string
}

/**
 * How many entries one list field of a profile keeps.
 *
 * Bound: the router previews each profile in its prompt and the daemon caps
 * a profile at 8 KiB; 32 model ids or tags is far past any real priority
 * list and keeps the field readable in one line of the roster.
 */
export const PROFILE_LIST_MAX = 32

/** Split a free-text list on commas/newlines: trimmed, de-duplicated, bounded. */
export function splitList(text: string): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const raw of text.split(/[,\n]/)) {
    const item = raw.trim()
    if (!item || seen.has(item.toLowerCase())) continue
    seen.add(item.toLowerCase())
    out.push(item)
    if (out.length >= PROFILE_LIST_MAX) break
  }
  return out
}

/** The form's view of a stored profile. */
export function formFromProfile(profile: unknown): ProfileForm {
  const p = (profile && typeof profile === 'object' ? profile : {}) as Record<string, unknown>
  const list = (v: unknown) => Array.isArray(v) ? v.filter(x => typeof x === 'string').join(', ') : ''
  return {
    models: list(p.model_priority),
    capabilities: list(p.capabilities),
    notes: typeof p.notes === 'string' ? p.notes : '',
  }
}

/**
 * Write the form back over `existing`, keeping every key the form does not
 * own (a profile may carry fields set elsewhere, e.g. the router's `role`).
 * An emptied field is removed rather than stored empty.
 */
export function profileFromForm(form: ProfileForm, existing: unknown = {}): Record<string, unknown> {
  const base = existing && typeof existing === 'object' && !Array.isArray(existing)
    ? { ...(existing as Record<string, unknown>) }
    : {}
  const set = (key: string, value: unknown, empty: boolean) => {
    if (empty) delete base[key]
    else base[key] = value
  }
  const models = splitList(form.models)
  const capabilities = splitList(form.capabilities)
  const notes = form.notes.trim()
  set('model_priority', models, models.length === 0)
  set('capabilities', capabilities, capabilities.length === 0)
  set('notes', notes, notes === '')
  return base
}
