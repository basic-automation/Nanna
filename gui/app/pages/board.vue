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
            {{ tab.label }}<span v-if="tab.id !== 'board'" class="ml-1 opacity-70">{{ tab.id === 'inbox' ? split.inbox.length : tab.id === 'upcoming' ? upcomingCount : roster.length }}</span>
          </button>
        </nav>
        <h1 class="text-sm font-semibold text-nui-fg">
          <span class="text-nui-muted">{{ VIEW_SUBTITLE[view] ?? boardName }}</span>
        </h1>
        <span class="min-w-0 flex-1" />
        <template v-if="view === 'board'">
          <select v-model="filters.assignee" data-testid="board-assignee-filter" :class="FILTER" aria-label="Assignee">
            <option value="">Anyone</option>
            <option v-for="member in assignableMembers" :key="member.id" :value="member.id">
              {{ member.name }}
            </option>
          </select>
          <select v-model="filters.label" data-testid="board-label-filter" :class="FILTER" aria-label="Label">
            <option value="">Any label</option>
            <option v-for="label in labels" :key="label" :value="label">#{{ label }}</option>
          </select>
          <select v-model="filters.priority" data-testid="board-priority-filter" :class="FILTER" aria-label="Priority">
            <option value="">Any priority</option>
            <option v-for="p in ['1', '2', '3', '4']" :key="p" :value="p">p{{ p }}</option>
          </select>
          <select v-model="filters.date" data-testid="board-date-filter" :class="FILTER" aria-label="Dates">
            <option value="">Any date</option>
            <option value="startable">Startable now</option>
            <option value="deferred">Deferred</option>
            <option value="overdue">Overdue</option>
            <option value="no_deadline">No deadline</option>
          </select>
          <button
            v-if="filtering(filters)"
            type="button"
            class="text-xs text-nui-muted hover:text-nui-fg"
            data-testid="board-clear-filters"
            @click="Object.assign(filters, NO_FILTERS)"
          >
            Clear
          </button>
        </template>
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
        v-if="view === 'inbox' || view === 'upcoming'"
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

      <!-- Members: who is on this board, and adding agents (decision 3) -->
      <BoardMembers
        v-else-if="view === 'members'"
        :roster="roster"
        :workspace-id="board.workspaceId"
        @changed="loadRoster"
      />

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
            :class="FIELD"
            :disabled="!editable"
            @change="assign(($event.target as HTMLSelectElement).value)"
          >
            <option value="" disabled>Unassigned — the router will pick</option>
            <option v-for="member in assignableMembers" :key="member.id" :value="member.id">
              {{ member.name }}
            </option>
          </select>
        </dd>
        <dt class="text-nui-muted">Priority</dt>
        <dd>
          <select
            :value="selected.priority"
            data-testid="card-priority"
            :class="FIELD"
            :disabled="!editable"
            @change="patchCard({ priority: Number(($event.target as HTMLSelectElement).value) })"
          >
            <option v-for="p in [1, 2, 3, 4]" :key="p" :value="p">p{{ p }}</option>
          </select>
        </dd>
        <dt class="text-nui-muted">Date</dt>
        <dd class="flex items-center gap-2">
          <!-- WebKitGTK draws an empty date input with today's date as its
               placeholder, which reads as a date that is set: show a
               button until there is one. -->
          <input
            v-if="dayOf(selected.due_at) || openDates.due_at"
            type="date"
            :value="dayOf(selected.due_at) ?? ''"
            data-testid="card-date"
            :class="FIELD"
            :disabled="!editable"
            title="Defers the card: it stays out of the inbox until this day"
            @change="patchCard({ due_at: ($event.target as HTMLInputElement).value })"
          >
          <button
            v-else
            type="button"
            class="text-xs text-nui-muted hover:text-nui-fg disabled:opacity-60"
            :data-testid="`card-date-set`"
            :disabled="!editable"
            @click="openDates.due_at = true"
          >
            None — set
          </button>
          <button
            v-if="dayOf(selected.due_at) && editable"
            type="button"
            class="shrink-0 text-xs text-nui-muted hover:text-nui-pink"
            :data-testid="`card-date-clear`"
            title="Clear"
            @click="openDates.due_at = false; patchCard({ due_at: '' })"
          >
            ×
          </button>
        </dd>
        <dt class="text-nui-muted">Deadline</dt>
        <dd class="flex items-center gap-2">
          <!-- WebKitGTK draws an empty date input with today's date as its
               placeholder, which reads as a date that is set: show a
               button until there is one. -->
          <input
            v-if="dayOf(selected.deadline_at) || openDates.deadline_at"
            type="date"
            :value="dayOf(selected.deadline_at) ?? ''"
            data-testid="card-deadline"
            :class="[FIELD, isOverdue(selected, today) && '!text-nui-pink']"
            :disabled="!editable"
            title="Must be done by; overdue is measured against this"
            @change="patchCard({ deadline_at: ($event.target as HTMLInputElement).value })"
          >
          <button
            v-else
            type="button"
            class="text-xs text-nui-muted hover:text-nui-fg disabled:opacity-60"
            :data-testid="`card-deadline-set`"
            :disabled="!editable"
            @click="openDates.deadline_at = true"
          >
            None — set
          </button>
          <button
            v-if="dayOf(selected.deadline_at) && editable"
            type="button"
            class="shrink-0 text-xs text-nui-muted hover:text-nui-pink"
            :data-testid="`card-deadline-clear`"
            title="Clear"
            @click="openDates.deadline_at = false; patchCard({ deadline_at: '' })"
          >
            ×
          </button>
        </dd>
        <dt class="text-nui-muted">Labels</dt>
        <dd>
          <input
            :value="selected.labels.join(', ')"
            data-testid="card-labels"
            :class="FIELD"
            :disabled="!editable"
            placeholder="comma, separated"
            @change="patchCard({ labels: splitList(($event.target as HTMLInputElement).value).map(l => l.replace(/^#/, '')) })"
          >
        </dd>
      </dl>
      <textarea
        :value="selected.description ?? ''"
        data-testid="card-description"
        rows="3"
        :class="[FIELD, 'resize-y']"
        :disabled="!editable"
        placeholder="Description — what done looks like, links, context"
        @change="patchCard({ description: ($event.target as HTMLTextAreaElement).value })"
      />

      <!-- Where the card sits: its parent, what it waits on, its sub-cards -->
      <section class="flex flex-col gap-2 text-xs" data-testid="card-relations">
        <p v-if="parentCard" class="text-nui-muted">
          Part of
          <button type="button" class="text-nui-accent hover:underline" @click="selectCard(parentCard.id)">
            #{{ parentCard.id }} {{ parentCard.title }}
          </button>
        </p>
        <div v-if="waitingOn.length" class="flex flex-col gap-1" data-testid="card-waiting-on">
          <p class="text-nui-yellow">Waiting on</p>
          <button
            v-for="dep in waitingOn"
            :key="dep.id"
            type="button"
            class="text-left text-nui-fg hover:underline"
            :class="dep.card && columnOf(dep.card) === 'done' && 'text-nui-muted line-through'"
            @click="selectCard(dep.id)"
          >
            #{{ dep.id }} {{ dep.card?.title ?? '(not on this board)' }}
            <span v-if="dep.card" class="text-nui-muted">— {{ memberName(dep.card.assignee, roster) }}</span>
          </button>
        </div>
        <div class="flex flex-col gap-1" data-testid="card-sub-cards">
          <p class="text-nui-muted">Sub-cards<span v-if="subCards.length">{{ ` · ${subCards.filter(c => columnOf(c) !== 'done').length}/${subCards.length} open` }}</span></p>
          <button
            v-for="sub in subCards"
            :key="sub.id"
            type="button"
            class="text-left text-nui-fg hover:underline"
            :class="columnOf(sub) === 'done' && 'text-nui-muted line-through'"
            @click="selectCard(sub.id)"
          >
            p{{ sub.priority }} {{ sub.title }} <span class="text-nui-muted">— {{ memberName(sub.assignee, roster) }}</span>
          </button>
          <form v-if="editable" class="flex gap-2" @submit.prevent="addSubCard">
            <input
              v-model="subCardText"
              data-testid="card-sub-card-add"
              :class="FIELD"
              placeholder="Add a sub-card — same tokens as quick-add"
              :disabled="saving"
            >
          </form>
        </div>
      </section>
      <div class="flex items-center gap-2">
        <button
          v-if="runAction"
          type="button"
          data-testid="card-run"
          class="rounded-lg px-3 py-1 text-xs disabled:opacity-50"
          :class="runAction === 'stop' ? 'bg-nui-pink text-nui-bg' : 'bg-white/10 text-nui-fg hover:bg-white/15'"
          :disabled="saving"
          :title="RUN_TITLE[runAction]"
          @click="toggleRun"
        >
          {{ RUN_LABEL[runAction] }}
        </button>
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
          <!-- Members post markdown (verdicts quote commands and output);
               renderMarkdown sanitizes, as it does for chat. -->
          <div class="board-post break-words text-xs text-nui-fg" v-html="renderMarkdown(post.content)" />
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
import { computed, inject, onMounted, onUnmounted, reactive, ref, watch, type Ref } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'
import { renderMarkdown } from '~/lib/markdown'
import {
  applyFilters, arrangeColumns, assignable, boardLabel, boardLabels, childCounts, filtering, NO_FILTERS, columnOf, dayOf, eventIsForBoard,
  isDeferred, isOverdue, memberName, postKindLabel, runActionFor, splitAssigned, splitList, todayUtc,
  type BoardCard, type BoardFilters, type BoardMember, type CardPost, type RunAction,
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

type View = 'board' | 'inbox' | 'upcoming' | 'members'
const VIEWS: ReadonlyArray<{ id: View, label: string }> = [
  { id: 'board', label: 'Board' },
  { id: 'inbox', label: 'Inbox' },
  { id: 'upcoming', label: 'Upcoming' },
  { id: 'members', label: 'Members' },
]
const view = ref<View>('board')
const VIEW_SUBTITLE: Partial<Record<View, string>> = {
  inbox: '— assigned to you, startable now',
  upcoming: '— assigned to you, by date',
}

const cards = ref<BoardCard[]>([])
const roster = ref<BoardMember[]>([])
const loading = ref(false)
const loadError = ref('')
const nested = ref(true)
const filters = reactive<BoardFilters>({ ...NO_FILTERS })
const FILTER = 'rounded-lg border border-white/10 bg-nui-bg px-2 py-1 text-xs text-nui-fg outline-none [color-scheme:dark] focus:ring-1 focus:ring-nui-accent'
const today = ref(todayUtc())

const assignableMembers = computed(() => assignable(roster.value))
const filtered = computed(() => applyFilters(cards.value, filters, today.value))
const labels = computed(() => boardLabels(cards.value))
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
    // The board this page shows, named by id: the daemon's own "active
    // workspace" is whatever any client last chose.
    const reply = await invoke<{ tasks?: BoardCard[] }>('list_tasks', {
      scope: board.value.scope, sessionId: null, includeClosed: true,
      workspaceId: board.value.workspaceId,
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
    const reply = await invoke('quick_add_card', {
      text, scope: board.value.scope, workspaceId: board.value.workspaceId,
    })
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
    void loadRunState(id)
  } catch (e) {
    console.error('Failed to load card:', e)
  }
}

/** Date inputs the human opened on a card that had no date yet. */
const openDates = reactive({ due_at: false, deadline_at: false })

function selectCard(id: number) {
  selectedId.value = id
  openDates.due_at = false
  openDates.deadline_at = false
  cardMessage.value = ''
  posts.value = []
  running.value = false
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

const FIELD = 'w-full rounded border border-white/10 bg-nui-bg px-2 py-0.5 text-xs text-nui-fg outline-none [color-scheme:dark] disabled:opacity-60'

/** The selected card's parent, if it has one on this board. */
const parentCard = computed(() => {
  const parentId = selected.value?.parent_id
  return parentId == null ? null : cards.value.find(card => card.id === parentId) ?? null
})

/** What the selected card waits on (its dependencies, scope-local by the store's rule). */
const waitingOn = computed(() => (selected.value?.depends_on ?? []).map(id => ({
  id,
  card: cards.value.find(card => card.id === id) ?? null,
})))

const subCards = computed(() => {
  const id = selected.value?.id
  return id == null ? [] : cards.value.filter(card => card.parent_id === id).sort((a, b) => a.priority - b.priority || a.id - b.id)
})

const subCardText = ref('')

async function addSubCard() {
  const parentId = selectedId.value
  const text = subCardText.value.trim()
  if (parentId === null || !text) return
  saving.value = true
  try {
    const reply = await invoke('quick_add_card', { text, scope: null, parentId })
    cardMessage.value = refusal(reply) ?? ''
    if (!refusal(reply)) subCardText.value = ''
    await Promise.all([loadCards(), loadAssigned()])
  } catch (e) {
    cardMessage.value = String(e)
  } finally {
    saving.value = false
  }
}

// ═══ The card's run (agent cards only) ═══
const running = ref(false)
const runAction = computed(() => selected.value ? runActionFor(selected.value, running.value, today.value) : null)
const RUN_LABEL: Record<NonNullable<RunAction>, string> = { start: 'Start now', resume: 'Resume', stop: 'Stop' }
const RUN_TITLE: Record<NonNullable<RunAction>, string> = {
  start: 'Its member starts on it now instead of waiting for its turn',
  resume: 'Its member picks the paused card back up',
  stop: 'Stop the run; the card stays with its member, paused, until resumed',
}

async function loadRunState(id: number) {
  try {
    const reply = await invoke<{ running?: boolean }>('card_run_status', { cardId: id })
    if (selectedId.value === id) running.value = reply?.running === true
  } catch (e) {
    console.error('Failed to read the card run:', e)
  }
}

async function toggleRun() {
  const id = selectedId.value
  const action = runAction.value
  if (id === null || !action) return
  saving.value = true
  try {
    const reply = await invoke(action === 'stop' ? 'stop_card_run' : 'start_card_run', { cardId: id })
    cardMessage.value = refusal(reply) ?? ''
    await Promise.all([loadCards(), loadCard(id), loadRoster()])
  } catch (e) {
    cardMessage.value = String(e)
  } finally {
    saving.value = false
  }
}

/** A closed card is history; its fields are shown, not edited. */
const editable = computed(() => !saving.value && selected.value !== null && columnOf(selected.value) !== 'done')

/** Apply one field change; the daemon validates it (a deadline before the date is refused). */
async function patchCard(patch: Record<string, unknown>) {
  const id = selectedId.value
  if (id === null) return
  saving.value = true
  try {
    const reply = await invoke('update_task', { id, patch })
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
  Object.assign(filters, NO_FILTERS)
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

<style scoped>
/* Thread posts are markdown: keep its blocks tight inside a post. */
.board-post :deep(p) { margin: 0 0 0.25rem; }
.board-post :deep(pre) { margin: 0.25rem 0; padding: 0.5rem; border-radius: 6px; background: rgb(255 255 255 / 0.05); overflow-x: auto; white-space: pre; }
.board-post :deep(code) { font-family: var(--font-nui); }
.board-post :deep(ul), .board-post :deep(ol) { margin: 0.25rem 0; padding-left: 1.25rem; }
.board-post :deep(a) { color: var(--color-nui-info); text-decoration: underline; }
</style>
