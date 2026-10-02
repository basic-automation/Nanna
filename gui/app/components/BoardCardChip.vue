<script setup lang="ts">
import {
  dayOf, initials, isDeferred, isOverdue, memberName, ROUTER_PREFIX,
  type BoardCard, type BoardMember,
} from '~/lib/board'

/** One card on the board, or in an Inbox/Upcoming list (P25 Stage 4). */
const props = defineProps<{
  card: BoardCard
  roster: readonly BoardMember[]
  /** The store's UTC day, for deferred/overdue colouring. */
  today: string
  selected?: boolean
  /** Open/total sub-cards, shown when sub-cards sit inside their parent. */
  children?: { open: number, total: number }
  /** The board's name, for lists that span boards. */
  boardName?: string
}>()

const emit = defineEmits<{ select: [id: number] }>()

const PRIORITY_CLASS = ['text-nui-pink', 'text-nui-yellow', 'text-nui-info', 'text-nui-muted']

function avatarClass(id: string | null): string {
  if (!id) return 'bg-nui-muted'
  if (id === 'human') return 'bg-nui-accent'
  if (id.startsWith(ROUTER_PREFIX)) return 'bg-nui-yellow'
  return 'bg-nui-pink'
}

/** In progress and its member is running: the only queue signal (decision 3). */
function isBusy(): boolean {
  const { card, roster } = props
  if (card.status !== 'in_progress' || !card.assignee) return false
  return roster.some(member => member.id === card.assignee && member.status === 'busy')
}
</script>

<template>
  <button
    type="button"
    class="flex w-full flex-col gap-2 rounded-lg bg-white/5 p-3 text-left hover:bg-white/10"
    :class="props.selected && 'ring-1 ring-nui-accent'"
    :data-testid="`board-card-${props.card.id}`"
    @click="emit('select', props.card.id)"
  >
    <div class="flex items-start gap-2">
      <span
        class="mt-0.5 shrink-0 text-xs font-semibold"
        :class="PRIORITY_CLASS[props.card.priority - 1] ?? 'text-nui-muted'"
        :title="`Priority ${props.card.priority}`"
      >p{{ props.card.priority }}</span>
      <span
        class="min-w-0 flex-1 break-words text-xs text-nui-fg"
        :class="props.card.status === 'cancelled' && 'line-through text-nui-muted'"
      >{{ props.card.title }}</span>
      <span v-if="props.card.blocked" class="shrink-0 text-[11px] text-nui-yellow" title="Waiting on another card">waiting</span>
    </div>
    <div v-if="props.card.labels.length" class="flex flex-wrap gap-1">
      <span
        v-for="label in props.card.labels"
        :key="label"
        class="rounded bg-nui-info/15 px-1.5 text-[11px] text-nui-info"
      >#{{ label }}</span>
    </div>
    <div class="flex flex-wrap items-center gap-2 text-[11px] text-nui-muted">
      <span class="flex items-center gap-1">
        <span
          class="flex size-5 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold text-nui-bg"
          :class="avatarClass(props.card.assignee)"
        >{{ initials(memberName(props.card.assignee, props.roster)) }}</span>
        <span>{{ memberName(props.card.assignee, props.roster) }}</span>
        <span v-if="isBusy()" class="text-nui-green" title="Working on it">●</span>
      </span>
      <span v-if="props.boardName" class="text-nui-accent" title="Board">{{ props.boardName }}</span>
      <span v-if="dayOf(props.card.due_at)" :class="isDeferred(props.card, props.today) && 'text-nui-yellow'" title="Date">
        {{ dayOf(props.card.due_at) }}
      </span>
      <span v-if="dayOf(props.card.deadline_at)" :class="isOverdue(props.card, props.today) ? 'text-nui-pink' : ''" title="Deadline">
        ⚑ {{ dayOf(props.card.deadline_at) }}
      </span>
      <span v-if="props.children" title="Open / all sub-cards">
        ↳ {{ props.children.open }}/{{ props.children.total }}
      </span>
    </div>
  </button>
</template>
