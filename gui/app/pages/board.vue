<template>
  <div class="flex h-full min-h-0 w-full gap-4 pr-4" data-testid="board-page">
    <!-- ═══ Board ═══ -->
    <section class="flex min-h-0 min-w-0 flex-1 flex-col gap-4">
      <header class="flex shrink-0 flex-wrap items-center gap-4">
        <nav class="flex items-center gap-1" aria-label="Board views">
          <button
            v-for="tab in VIEWS"
            :key="tab.id"
            type="button"
            class="rounded-lg px-3 py-1 text-xs"
            :class="view === tab.id ? 'bg-nui-accent text-nui-fg' : 'text-nui-muted hover:bg-white/5'"
            :data-testid="`board-view-${tab.id}`"
            :aria-pressed="view === tab.id"
            @click="view = tab.id"
          >
            {{ tab.label }}<span v-if="tab.id !== 'board'" class="ml-1 opacity-70">{{ tab.id === 'inbox' ? split.inbox.length : upcomingCount }}</span>
          </button>
        </nav>
        <h1 class="text-sm font-semibold text-nui-fg">
          <span class="text-nui-muted">{{ view === 'board' ? boardName : view === 'inbox' ? '— assigned to you, startable now' : '— assigned to you, by date' }}</span>
        </h1>
        <span class="min-w-0 flex-1" />
        <label v-if="view === 'board'" class="flex items-center gap-2 text-xs text-nui-muted">
          <span>Assignee</span>
          <select
            v-model="assigneeFilter"
            data-testid="board-assignee-filter"
            class="rounded-lg border border-white/10 bg-nui-bg px-2 py-1 text-xs text-nui-fg outline-none [color-scheme:dark] focus:ring-1 focus:ring-nui-accent"
          >
            <option value="">Everyone</option>
            <option v-for="member in assignableMembers" :key="member.id" :value="member.id">
              {{ member.name }}
            </option>
          </select>
        </label>
        <button
          v-if="view === 'board'"
          type="button"
          data-testid="board-nested-toggle"
          class="rounded-lg bg-white/5 px-3 py-1 text-xs text-nui-fg hover:bg-white/10"
          :aria-pressed="nested"
          @click="nested = !nested"
        >
          {{ nested ? 'Sub-cards: inside parents' : 'Sub-cards: on the board' }}
        </button>
      </header>

      <!-- Quick-add: the board's only free-text entry (P25 decision 1) -->
      <form class="flex shrink-0 flex-col gap-1" @submit.prevent="submitQuickAdd">
        <div class="flex items-center gap-2 rounded-lg bg-white/5 px-4 py-2 focus-within:ring-1 focus-within:ring-nui-accent">
          <NuiIcon name="add" :size="16" class="text-nui-muted" />
          <input
            v-model="quickAddText"
            data-testid="board-quick-add"
            type="text"
            class="min-w-0 flex-1 bg-transparent text-xs text-nui-fg outline-none placeholder:text-nui-muted"
            placeholder="Add a card — #label p1 @member friday {deadline}"
            :disabled="adding"
            aria-label="Quick add a card"
          >
          <button
            type="submit"
            class="text-xs text-nui-accent disabled:text-nui-muted"
            :disabled="adding || !quickAddText.trim()"
          >
            Add
          </button>
        </div>
        <p v-if="quickAddError" class="px-4 text-xs text-nui-pink" data-testid="board-quick-add-error">
          {{ quickAddError }}
        </p>
      </form>

      <p v-if="loadError" class="text-xs text-nui-pink">{{ loadError }}</p>

      <!-- Inbox / Upcoming: one member's cards across every board (decision 11) -->
      <div
        v-if="view !== 'board'"
        class="nui-scroll flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto"
        :data-testid="`board-${view}`"
      >
        <template v-if="view === 'inbox'">
          <div class="flex max-w-3xl flex-col gap-2">
            <BoardCardChip
              v-for="card in split.inbox"
              :key="card.id"
              :card="card"
              :roster="roster"
              :today="assignedToday"
              :selected="selectedId === card.id"
              :board-name="boardLabel(card, workspaces)"
              @select="selectCard"
            />
          </div>
          <p v-if="split.inbox.length === 0" class="text-xs text-nui-muted">Nothing assigned to you is startable now.</p>
        </template>
        <template v-else>
          <section v-for="group in split.upcoming" :key="group.day" class="flex max-w-3xl flex-col gap-2">
            <p class="text-xs font-semibold text-nui-fg">{{ group.day }}</p>
            <BoardCardChip
              v-for="card in group.cards"
              :key="card.id"
              :card="card"
              :roster="roster"
              :today="assignedToday"
              :selected="selectedId === card.id"
              :board-name="boardLabel(card, workspaces)"
              @select="selectCard"
            />
          </section>
          <p v-if="split.upcoming.length === 0" class="text-xs text-nui-muted">Nothing assigned to you is dated later.</p>
        </template>
      </div>

      <!-- Columns -->
      <div v-else class="nui-scroll grid min-h-0 flex-1 grid-cols-4 gap-4 overflow-x-auto" data-testid="board-columns">
        <div
          v-for="column in columns"
          :key="column.id"
          class="flex min-h-0 min-w-48 flex-col gap-2 rounded-lg bg-white/[0.03] p-2"
          :data-testid="`board-column-${column.id}`"
        >
          <p class="flex items-center gap-2 px-2 pt-1 text-xs text-nui-muted">
            <span class="font-semibold text-nui-fg">{{ column.label }}</span>
            <span>{{ column.cards.length }}</span>
          </p>
          <div class="nui-scroll flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto">
            <BoardCardChip
              v-for="card in column.cards"
              :key="card.id"
              :card="card"
              :roster="roster"
              :today="today"
              :selected="selectedId === card.id"
              :children="nested ? children.get(card.id) : undefined"
              @select="selectCard"
            />
            <p v-if="column.cards.length === 0" class="px-2 py-4 text-center text-xs text-nui-muted">
              {{ column.id === 'todo' && cards.length === 0 && !loading ? 'No cards yet — add one above' : 'Nothing here' }}
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- ═══ Card view: the thread lives here (P25 decision 2) ═══ -->
    <aside
      v-if="selected"
      class="nui-scroll flex w-96 shrink-0 flex-col gap-4 overflow-y-auto rounded-lg bg-white/[0.03] p-4"
      data-testid="board-card-view"
    >
      <div class="flex items-start gap-2">
        <h2 class="min-w-0 flex-1 break-words text-sm font-semibold text-nui-fg">{{ selected.title }}</h2>
        <NuiIconButton icon="close" label="Close card" @click="selectedId = null" />
      </div>
      <dl class="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-xs">
        <dt class="text-nui-muted">Status</dt>
        <dd class="text-nui-fg">{{ statusText(selected) }}</dd>
        <dt class="text-nui-muted">Assignee</dt>
        <dd>
          <select
            :value="selected.assignee ?? ''"
            data-testid="card-assignee"
            class="w-full rounded border border-white/10 bg-nui-bg px-2 py-0.5 text-xs text-nui-fg outline-none [color-scheme:dark]"
            :disabled="saving || columnOf(selected) === 'done'"
            @change="assign(($event.target as HTMLSelectElement).value)"
          >
            <option value="" disabled>Unassigned — the router will pick</option>
            <option v-for="member in assignableMembers" :key="member.id" :value="member.id">
              {{ member.name }}
            </option>
          </select>
        </dd>
        <dt class="text-nui-muted">Priority</dt>
        <dd class="text-nui-fg">p{{ selected.priority }}</dd>
        <template v-if="dayOf(selected.due_at)">
          <dt class="text-nui-muted">Date</dt>
          <dd class="text-nui-fg">{{ dayOf(selected.due_at) }}</dd>
        </template>
        <template v-if="dayOf(selected.deadline_at)">
          <dt class="text-nui-muted">Deadline</dt>
          <dd :class="isOverdue(selected, today) ? 'text-nui-pink' : 'text-nui-fg'">{{ dayOf(selected.deadline_at) }}</dd>
        </template>
        <template v-if="selected.labels.length">
          <dt class="text-nui-muted">Labels</dt>
          <dd class="text-nui-info">{{ selected.labels.map(l => '#' + l).join(' ') }}</dd>
        </template>
      </dl>
      <p v-if="selected.description" class="whitespace-pre-wrap break-words text-xs text-nui-fg">{{ selected.description }}</p>
      <div class="flex items-center gap-2">
        <button
          v-if="columnOf(selected) !== 'done'"
          type="button"
          data-testid="card-done"
          class="rounded-lg bg-nui-accent px-3 py-1 text-xs text-nui-fg disabled:opacity-50"
          :disabled="saving"
          @click="markDone"
        >
          Mark done
        </button>
        <p v-if="cardMessage" class="text-xs text-nui-yellow">{{ cardMessage }}</p>
      </div>

      <section class="flex flex-col gap-2" data-testid="card-thread">
        <p class="text-xs font-semibold text-nui-fg">Thread</p>
        <article
          v-for="post in posts"
          :key="post.id"
          class="flex flex-col gap-1 rounded-lg bg-white/5 p-2"
        >
          <p class="flex items-center gap-2 text-[11px] text-nui-muted">
            <span class="text-nui-fg">{{ memberName(post.author_member_id ?? post.author, roster) }}</span>
            <span :class="post.kind === 'question' ? 'text-nui-yellow' : post.kind === 'verdict' ? 'text-nui-green' : ''">
              {{ postKindLabel(post.kind) }}
            </span>
            <span class="min-w-0 flex-1" />
            <span>{{ post.created_at.slice(0, 16) }}</span>
          </p>
          <p class="whitespace-pre-wrap break-words text-xs text-nui-fg">{{ post.content }}</p>
        </article>
        <p v-if="posts.length === 0" class="text-xs text-nui-muted">No posts yet.</p>
        <form class="flex flex-col gap-2" @submit.prevent="submitPost">
          <textarea
            v-model="postText"
            data-testid="card-post"
            rows="3"
            class="w-full resize-y rounded-lg bg-white/5 p-2 text-xs text-nui-fg outline-none placeholder:text-nui-muted focus:ring-1 focus:ring-nui-accent"
            placeholder="Post on this card…"
            :disabled="posting"
            @keydown.enter.ctrl.prevent="submitPost"
            @keydown.enter.meta.prevent="submitPost"
          />
          <button
            type="submit"
            class="self-end rounded-lg bg-white/5 px-3 py-1 text-xs text-nui-fg hover:bg-white/10 disabled:opacity-50"
            :disabled="posting || !postText.trim()"
          >
            Post
          </button>
        </form>
      </section>
    </aside>
  </div>
</template>

<script setup lang="ts">
import { computed, inject, onMounted, onUnmounted, ref, watch, type Ref } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'
import {
  arrangeColumns, assignable, boardLabel, childCounts, columnOf, dayOf, eventIsForBoard,
  isDeferred, isOverdue, memberName, postKindLabel, splitAssigned, todayUtc,
  type BoardCard, type BoardMember, type CardPost,
} from '~/lib/board'

interface WorkspaceInfo { id: string, name: string, path: string }

const activeWorkspace = inject<Ref<WorkspaceInfo | null>>('activeWorkspace', ref(null))

/** Which board: the open workspace's, else the global one. */
const board = computed(() => activeWorkspace.value
  ? { scope: 'workspace' as const, workspaceId: activeWorkspace.value.id }
  : { scope: 'global' as const, workspaceId: null })
const boardName = computed(() => activeWorkspace.value
  ? `— ${activeWorkspace.value.name || activeWorkspace.value.path}`
  : '— Global')

type View = 'board' | 'inbox' | 'upcoming'
const VIEWS: ReadonlyArray<{ id: View, label: string }> = [
  { id: 'board', label: 'Board' },
  { id: 'inbox', label: 'Inbox' },
  { id: 'upcoming', label: 'Upcoming' },
]
const view = ref<View>('board')

const cards = ref<BoardCard[]>([])
const roster = ref<BoardMember[]>([])
const loading = ref(false)
const loadError = ref('')
const nested = ref(true)
const assigneeFilter = ref('')
const today = ref(todayUtc())

const assignableMembers = computed(() => assignable(roster.value))
const filtered = computed(() => assigneeFilter.value
  ? cards.value.filter(card => card.assignee === assigneeFilter.value)
  : cards.value)
const columns = computed(() => arrangeColumns(filtered.value, nested.value))
const children = computed(() => childCounts(cards.value))

/** A daemon reply that refused the action carries `error` + `message`. */
function refusal(reply: unknown): string | null {
  const r = reply as { error?: string, message?: string } | null
  return r?.error ? (r.message || r.error) : null
}

async function loadCards() {
  loading.value = true
  try {
    const reply = await invoke<{ tasks?: BoardCard[] }>('list_tasks', {
      scope: board.value.scope, sessionId: null, includeClosed: true,
    })
    const why = refusal(reply)
    loadError.value = why ?? ''
    if (!why) cards.value = reply.tasks ?? []
    today.value = todayUtc()
  } catch (e) {
    loadError.value = `Could not load the board: ${e}`
  } finally {
    loading.value = false
  }
}

// ═══ Inbox / Upcoming ═══
const assigned = ref<BoardCard[]>([])
const assignedToday = ref(todayUtc())
const workspaces = ref<WorkspaceInfo[]>([])
const split = computed(() => splitAssigned(assigned.value, assignedToday.value))
const upcomingCount = computed(() => split.value.upcoming.reduce((n, day) => n + day.cards.length, 0))

async function loadAssigned() {
  try {
    const reply = await invoke<{ cards?: BoardCard[], today?: string }>('list_assigned_cards', { memberId: null })
    if (refusal(reply)) return
    assigned.value = reply.cards ?? []
    if (reply.today) assignedToday.value = reply.today
  } catch (e) {
    console.error('Failed to load assigned cards:', e)
  }
}

async function loadWorkspaces() {
  try {
    workspaces.value = await invoke<WorkspaceInfo[]>('list_workspaces')
  } catch (e) {
    console.error('Failed to load workspaces:', e)
  }
}

async function loadRoster() {
  try {
    const reply = await invoke<{ members?: BoardMember[] }>('list_members', {
      workspaceId: board.value.workspaceId,
    })
    if (!refusal(reply)) roster.value = reply.members ?? []
  } catch (e) {
    console.error('Failed to load the board roster:', e)
  }
}

/** A member is busy on this card: it is in progress and its member works. */
function isBusy(card: BoardCard): boolean {
  if (card.status !== 'in_progress' || !card.assignee) return false
  return roster.value.some(member => member.id === card.assignee && member.status === 'busy')
}

function statusText(card: BoardCard): string {
  if (card.status === 'done') return 'Done'
  if (card.status === 'cancelled') return 'Cancelled'
  if (card.blocked) return 'Waiting on another card'
  if (card.status === 'in_progress') return isBusy(card) ? 'In progress — being worked' : 'In progress — paused'
  return isDeferred(card, today.value) ? `Deferred until ${dayOf(card.due_at)}` : 'To do'
}

// ═══ Quick-add ═══
const quickAddText = ref('')
const quickAddError = ref('')
const adding = ref(false)

async function submitQuickAdd() {
  const text = quickAddText.value.trim()
  if (!text || adding.value) return
  adding.value = true
  quickAddError.value = ''
  try {
    const reply = await invoke('quick_add_card', { text, scope: board.value.scope })
    const why = refusal(reply)
    if (why) {
      quickAddError.value = why
      return
    }
    quickAddText.value = ''
    await Promise.all([loadCards(), loadAssigned()])
  } catch (e) {
    quickAddError.value = String(e)
  } finally {
    adding.value = false
  }
}

// ═══ Card view ═══
const selectedId = ref<number | null>(null)
const selectedDetail = ref<BoardCard | null>(null)
const posts = ref<CardPost[]>([])
const postText = ref('')
const posting = ref(false)
const saving = ref(false)
const cardMessage = ref('')

/** The card as the list last read it, refreshed by its own `task.get`. */
const selected = computed(() => {
  if (selectedId.value === null) return null
  return selectedDetail.value?.id === selectedId.value
    ? selectedDetail.value
    : cards.value.find(card => card.id === selectedId.value) ?? null
})

async function loadCard(id: number) {
  try {
    const reply = await invoke<{ task?: BoardCard, notes?: CardPost[] }>('get_card', { id })
    if (refusal(reply) || selectedId.value !== id) return
    selectedDetail.value = reply.task ?? null
    posts.value = reply.notes ?? []
  } catch (e) {
    console.error('Failed to load card:', e)
  }
}

function selectCard(id: number) {
  selectedId.value = id
  cardMessage.value = ''
  posts.value = []
  selectedDetail.value = null
  void loadCard(id)
}

async function submitPost() {
  const id = selectedId.value
  const content = postText.value.trim()
  if (id === null || !content || posting.value) return
  posting.value = true
  try {
    const reply = await invoke('post_on_card', { id, content })
    const why = refusal(reply)
    if (why) {
      cardMessage.value = why
      return
    }
    postText.value = ''
    await loadCard(id)
  } catch (e) {
    cardMessage.value = String(e)
  } finally {
    posting.value = false
  }
}

async function assign(memberId: string) {
  const id = selectedId.value
  if (id === null || !memberId) return
  saving.value = true
  try {
    const reply = await invoke('update_task', { id, patch: { assignee: memberId } })
    cardMessage.value = refusal(reply) ?? ''
    await Promise.all([loadCards(), loadAssigned(), loadCard(id)])
  } catch (e) {
    cardMessage.value = String(e)
  } finally {
    saving.value = false
  }
}

async function markDone() {
  const id = selectedId.value
  if (id === null) return
  saving.value = true
  try {
    const reply = await invoke<{ done?: boolean, verdict?: string }>('complete_task', { id, workdir: null })
    // Done is a verdict (P15): a failed acceptance check leaves the card open.
    cardMessage.value = refusal(reply) ?? (reply.done === false ? `Not done: ${reply.verdict ?? 'the check failed'}` : '')
    await Promise.all([loadCards(), loadAssigned(), loadCard(id)])
  } catch (e) {
    cardMessage.value = String(e)
  } finally {
    saving.value = false
  }
}

// ═══ Live refresh: the daemon announces every card change on the bus ═══
let refreshTimer: ReturnType<typeof setTimeout> | null = null
const unlisteners: UnlistenFn[] = []

/** Coalesce a burst of events (a router decision writes several) into one read. */
function scheduleRefresh() {
  if (refreshTimer) clearTimeout(refreshTimer)
  refreshTimer = setTimeout(() => {
    refreshTimer = null
    void loadCards()
    void loadAssigned()
    // A member's busy/idle flips with its run, which moves cards but is not
    // itself a roster change — read both.
    void loadRoster()
    if (selectedId.value !== null) void loadCard(selectedId.value)
  }, 250)
}

watch(board, () => {
  selectedId.value = null
  assigneeFilter.value = ''
  void loadCards()
  void loadRoster()
})

watch(view, (next) => {
  if (next !== 'board') {
    void loadAssigned()
    void loadWorkspaces()
  }
})

onMounted(async () => {
  void loadCards()
  void loadRoster()
  void loadAssigned()
  try {
    unlisteners.push(await listen<{ scope?: string, scope_id?: string | null }>('board-event', (event) => {
      // Inbox and Upcoming span every board, so any card change may move them.
      if (view.value !== 'board' || eventIsForBoard(event.payload ?? {}, board.value)) scheduleRefresh()
    }))
    unlisteners.push(await listen('members-changed', () => { void loadRoster() }))
  } catch (e) {
    console.error('Failed to subscribe to board events:', e)
  }
})

onUnmounted(() => {
  if (refreshTimer) clearTimeout(refreshTimer)
  for (const unlisten of unlisteners) unlisten()
})
</script>
