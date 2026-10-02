import { describe, expect, it } from 'vitest'
import {
  arrangeColumns,
  boardLabel,
  splitAssigned,
  splitList,
  applyFilters,
  boardNoticeFor,
  boardLabels,
  filtering,
  NO_FILTERS,
  type BoardFilters,
  formFromProfile,
  profileFromForm,
  PROFILE_LIST_MAX,
  assignable,
  childCounts,
  columnOf,
  DONE_SHOWN_MAX,
  eventIsForBoard,
  initials,
  isDeferred,
  isOverdue,
  memberName,
  todayUtc,
  type BoardCard,
  type BoardMember,
} from '../../app/lib/board'

let nextId = 1
function card(fields: Partial<BoardCard> = {}): BoardCard {
  const id = fields.id ?? nextId++
  return {
    id,
    parent_id: null,
    scope: 'global',
    scope_id: null,
    title: `card ${id}`,
    description: null,
    status: 'pending',
    priority: 3,
    labels: [],
    due_at: null,
    deadline_at: null,
    assignee: null,
    blocked: false,
    sort_order: id,
    created_at: '2026-10-01 10:00:00',
    ...fields,
  }
}

const ROSTER: BoardMember[] = [
  { id: 'human', name: 'You', avatar: null, kind: 'human', status: 'idle' },
  { id: 'router:global', name: 'Task Manager', avatar: null, kind: 'agent', status: 'idle' },
  { id: 'agent:builder', name: 'Builder', avatar: null, kind: 'agent', status: 'busy' },
]

describe('columnOf', () => {
  it('puts closed cards in done whatever else they say', () => {
    expect(columnOf(card({ status: 'done', blocked: true }))).toBe('done')
    expect(columnOf(card({ status: 'cancelled' }))).toBe('done')
  })

  it('a blocked open card waits, even mid-run', () => {
    expect(columnOf(card({ blocked: true }))).toBe('blocked')
    expect(columnOf(card({ status: 'in_progress', blocked: true }))).toBe('blocked')
  })

  it('splits the rest by stored status', () => {
    expect(columnOf(card({ status: 'in_progress' }))).toBe('in_progress')
    expect(columnOf(card())).toBe('todo')
  })
})

describe('arrangeColumns', () => {
  it('orders open columns by priority, then board order, then age', () => {
    const cards = [
      card({ id: 10, priority: 3, sort_order: 1 }),
      card({ id: 11, priority: 1, sort_order: 9 }),
      card({ id: 12, priority: 3, sort_order: 0 }),
      card({ id: 13, priority: 3, sort_order: 0 }),
    ]
    const todo = arrangeColumns(cards, false).find(c => c.id === 'todo')!
    expect(todo.cards.map(c => c.id)).toEqual([11, 12, 13, 10])
  })

  it('shows the newest finished first and bounds the done column', () => {
    const done = Array.from({ length: DONE_SHOWN_MAX + 5 }, (_, i) =>
      card({ status: 'done', updated_at: `2026-09-${String((i % 28) + 1).padStart(2, '0')} 10:00:${String(i % 60).padStart(2, '0')}` }))
    const column = arrangeColumns(done, false).find(c => c.id === 'done')!
    expect(column.cards).toHaveLength(DONE_SHOWN_MAX)
    const stamps = column.cards.map(c => c.updated_at!)
    expect([...stamps].sort().reverse()).toEqual(stamps)
  })

  it('nested hides sub-cards; flat shows them', () => {
    const parent = card({ id: 100 })
    const child = card({ id: 101, parent_id: 100 })
    const count = (nested: boolean) => arrangeColumns([parent, child], nested)
      .reduce((total, column) => total + column.cards.length, 0)
    expect(count(true)).toBe(1)
    expect(count(false)).toBe(2)
  })

  it('always returns the four columns in board order', () => {
    expect(arrangeColumns([], true).map(c => c.id)).toEqual(['todo', 'blocked', 'in_progress', 'done'])
  })
})

describe('childCounts', () => {
  it('counts open and total sub-cards per parent', () => {
    const counts = childCounts([
      card({ id: 200 }),
      card({ parent_id: 200 }),
      card({ parent_id: 200, status: 'done' }),
      card({ parent_id: 200, status: 'cancelled' }),
    ])
    expect(counts.get(200)).toEqual({ open: 1, total: 3 })
    expect(counts.has(201)).toBe(false)
  })
})

describe('members', () => {
  it('names members from the roster, falling back to the id', () => {
    expect(memberName('agent:builder', ROSTER)).toBe('Builder')
    expect(memberName('agent:gone', ROSTER)).toBe('agent:gone')
    expect(memberName(null, ROSTER)).toBe('Unassigned')
  })

  it('never offers the router as an assignee', () => {
    expect(assignable(ROSTER).map(m => m.id)).toEqual(['human', 'agent:builder'])
  })

  it('makes up to two initials', () => {
    expect(initials('Ada Lovelace')).toBe('AL')
    expect(initials('Ada Augusta King Lovelace')).toBe('AL')
    expect(initials('builder')).toBe('B')
    expect(initials('  ')).toBe('?')
    expect(initials('ミク')).toBe('ミ')
  })
})

describe('dates', () => {
  const today = '2026-10-07'

  it('overdue reads the deadline, and only on open cards', () => {
    expect(isOverdue(card({ deadline_at: '2026-10-06' }), today)).toBe(true)
    expect(isOverdue(card({ deadline_at: '2026-10-07' }), today)).toBe(false)
    expect(isOverdue(card({ deadline_at: '2026-10-06', status: 'done' }), today)).toBe(false)
    expect(isOverdue(card({ due_at: '2026-10-01' }), today), 'a passed date is not late').toBe(false)
  })

  it('a future date defers a card; today or a timestamp today does not', () => {
    expect(isDeferred(card({ due_at: '2026-10-08' }), today)).toBe(true)
    expect(isDeferred(card({ due_at: '2026-10-07T23:00:00Z' }), today)).toBe(false)
    expect(isDeferred(card(), today)).toBe(false)
  })

  it('today is the UTC day the store uses', () => {
    expect(todayUtc(new Date('2026-10-07T23:30:00-05:00'))).toBe('2026-10-08')
  })
})

describe('eventIsForBoard', () => {
  it('matches a workspace board by its id and the global board by scope', () => {
    const ws = { scope: 'workspace' as const, workspaceId: 'ws-1' }
    const global = { scope: 'global' as const, workspaceId: null }
    expect(eventIsForBoard({ scope: 'workspace', scope_id: 'ws-1' }, ws)).toBe(true)
    expect(eventIsForBoard({ scope: 'workspace', scope_id: 'ws-2' }, ws)).toBe(false)
    expect(eventIsForBoard({ scope: 'global' }, ws)).toBe(false)
    expect(eventIsForBoard({ scope: 'global', scope_id: null }, global)).toBe(true)
    expect(eventIsForBoard({ scope: 'session', scope_id: 's' }, global)).toBe(false)
  })
})

describe('splitAssigned', () => {
  const today = '2026-10-07'

  it('puts undated and due-today-or-past cards in the inbox, later ones by day', () => {
    const cards = [
      card({ id: 1, due_at: '2026-10-09' }),
      card({ id: 2 }),
      card({ id: 3, due_at: '2026-10-07T09:00:00Z', priority: 1 }),
      card({ id: 4, due_at: '2026-10-01' }),
      card({ id: 5, due_at: '2026-10-08' }),
      card({ id: 6, due_at: '2026-10-09', priority: 1 }),
    ]
    const { inbox, upcoming } = splitAssigned(cards, today)
    expect(inbox.map(c => c.id)).toEqual([3, 2, 4])
    expect(upcoming.map(d => d.day)).toEqual(['2026-10-08', '2026-10-09'])
    expect(upcoming[1]?.cards.map(c => c.id)).toEqual([6, 1])
  })

  it('names the board a card is on', () => {
    const workspaces = [{ id: 'ws-1', name: 'Nanna', path: '/src/nanna' }, { id: 'ws-2', name: '', path: '/src/x' }]
    expect(boardLabel(card({ scope: 'global' }), workspaces)).toBe('Global')
    expect(boardLabel(card({ scope: 'workspace', scope_id: 'ws-1' }), workspaces)).toBe('Nanna')
    expect(boardLabel(card({ scope: 'workspace', scope_id: 'ws-2' }), workspaces)).toBe('/src/x')
    expect(boardLabel(card({ scope: 'workspace', scope_id: 'gone' }), workspaces)).toBe('gone')
  })
})

describe('member profiles', () => {
  it('splits a list on commas and newlines, trimmed, deduplicated and bounded', () => {
    expect(splitList(' qwen3.5:9b, claude-sonnet-5\n qwen3.5:9b ,, ')).toEqual(['qwen3.5:9b', 'claude-sonnet-5'])
    expect(splitList('Rust, rust, RUST')).toEqual(['Rust'])
    expect(splitList(Array.from({ length: 40 }, (_, i) => `m${i}`).join(','))).toHaveLength(PROFILE_LIST_MAX)
  })

  it('round-trips the fields it owns and keeps the ones it does not', () => {
    const stored = { role: 'router', model_priority: ['a', 'b'], capabilities: ['rust'], notes: 'hi', cost: 3 }
    const form = formFromProfile(stored)
    expect(form).toEqual({ models: 'a, b', capabilities: 'rust', notes: 'hi' })
    const written = profileFromForm({ ...form, models: 'c' }, stored)
    expect(written).toEqual({ role: 'router', model_priority: ['c'], capabilities: ['rust'], notes: 'hi', cost: 3 })
  })

  it('removes an emptied field instead of storing it empty', () => {
    expect(profileFromForm({ models: ' ', capabilities: '', notes: '' }, { model_priority: ['a'], notes: 'x' })).toEqual({})
  })

  it('reads a missing or malformed profile as an empty form', () => {
    expect(formFromProfile(null)).toEqual({ models: '', capabilities: '', notes: '' })
    expect(formFromProfile({ model_priority: 'not a list', notes: 3 })).toEqual({ models: '', capabilities: '', notes: '' })
  })
})

describe('board filters', () => {
  const today = '2026-10-07'
  const cards = [
    card({ id: 1, assignee: 'human', labels: ['Rust'], priority: 1, deadline_at: '2026-10-01' }),
    card({ id: 2, assignee: 'agent:builder', labels: ['docs'], due_at: '2026-10-20' }),
    card({ id: 3, labels: ['rust', 'docs'], priority: 1 }),
  ]
  const ids = (filters: Partial<BoardFilters>) =>
    applyFilters(cards, { ...NO_FILTERS, ...filters }, today).map(c => c.id)

  it('passes everything with no filter set', () => {
    expect(ids({})).toEqual([1, 2, 3])
    expect(filtering({ ...NO_FILTERS })).toBe(false)
    expect(filtering({ ...NO_FILTERS, label: 'x' })).toBe(true)
  })

  it('narrows by each field, and they combine', () => {
    expect(ids({ assignee: 'human' })).toEqual([1])
    expect(ids({ label: 'RUST' })).toEqual([1, 3])
    expect(ids({ priority: '1' })).toEqual([1, 3])
    expect(ids({ label: 'docs', priority: '1' })).toEqual([3])
  })

  it('date filters read the deadline for overdue and the date for deferral', () => {
    expect(ids({ date: 'overdue' })).toEqual([1])
    expect(ids({ date: 'deferred' })).toEqual([2])
    expect(ids({ date: 'startable' })).toEqual([1, 3])
    expect(ids({ date: 'no_deadline' })).toEqual([2, 3])
  })

  it('lists the board\'s labels once each, sorted', () => {
    expect(boardLabels(cards)).toEqual(['docs', 'Rust'])
  })
})

describe('boardNoticeFor', () => {
  const mine = (fields: Partial<BoardCard> = {}) => card({ assignee: 'human', title: 'Fix the build', ...fields })

  it('tells me when my card\'s date arrives or its deadline passes', () => {
    expect(boardNoticeFor({ kind: 'due', task_id: 1, actor: 'sweep' }, mine())?.title).toBe('Ready to start: Fix the build')
    const overdue = boardNoticeFor({ kind: 'overdue', task_id: 1, actor: 'sweep' }, mine({ deadline_at: '2026-10-01' }))
    expect(overdue).toEqual({ type: 'warning', title: 'Overdue: Fix the build', summary: 'The deadline 2026-10-01 has passed.' })
  })

  it('tells me when someone else puts a card — or a question — in my hands', () => {
    expect(boardNoticeFor({ kind: 'created', task_id: 1, actor: 'router:global' }, mine())?.title)
      .toBe('New card for you from router:global')
    expect(boardNoticeFor({ kind: 'created', task_id: 1, actor: 'agent:builder' }, mine({ labels: ['clarification'] }))?.title)
      .toBe('A question for you from agent:builder')
  })

  it('stays quiet about my own writes, others\' cards, closed cards and other kinds', () => {
    expect(boardNoticeFor({ kind: 'created', task_id: 1, actor: 'gui' }, mine())).toBeNull()
    expect(boardNoticeFor({ kind: 'due', task_id: 1 }, mine({ assignee: 'agent:builder' }))).toBeNull()
    expect(boardNoticeFor({ kind: 'overdue', task_id: 1 }, mine({ status: 'done' }))).toBeNull()
    expect(boardNoticeFor({ kind: 'posted', task_id: 1, actor: 'agent:builder' }, mine())).toBeNull()
  })
})
