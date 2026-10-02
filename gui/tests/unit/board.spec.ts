import { describe, expect, it } from 'vitest'
import {
  arrangeColumns,
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
