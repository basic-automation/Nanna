<script setup lang="ts">
import { ref } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import {
  formFromProfile, initials, profileFromForm, ROUTER_PREFIX,
  type BoardMember, type ProfileForm,
} from '~/lib/board'

/**
 * A board's members (P25 decision 3): the human, the agents, and the router.
 * Agents are added and edited here; the router takes no work but its model
 * list is set here too. Every write goes to the daemon, which announces
 * `members-changed` and the board re-reads the roster.
 */
const props = defineProps<{
  roster: readonly BoardMember[]
  /** The board's workspace; `null` is the global board. */
  workspaceId: string | null
}>()

const emit = defineEmits<{ changed: [] }>()

const blankForm = (): ProfileForm => ({ models: '', capabilities: '', notes: '' })

const newName = ref('')
const newForm = ref<ProfileForm>(blankForm())
const newPersonal = ref(false)
const message = ref('')
const busy = ref(false)

const editingId = ref<string | null>(null)
const editName = ref('')
const editForm = ref<ProfileForm>(blankForm())

function refusal(reply: unknown): string | null {
  const r = reply as { error?: string, message?: string } | null
  return r?.error ? (r.message || r.error) : null
}

async function run(action: () => Promise<unknown>): Promise<boolean> {
  busy.value = true
  message.value = ''
  try {
    const why = refusal(await action())
    if (why) {
      message.value = why
      return false
    }
    emit('changed')
    return true
  } catch (e) {
    message.value = String(e)
    return false
  } finally {
    busy.value = false
  }
}

async function addAgent() {
  const name = newName.value.trim()
  if (!name) return
  const ok = await run(() => invoke('create_member', {
    name,
    workspaceId: newPersonal.value ? null : props.workspaceId,
    personal: newPersonal.value,
    profile: profileFromForm(newForm.value),
  }))
  if (ok) {
    newName.value = ''
    newForm.value = blankForm()
    newPersonal.value = false
  }
}

function startEdit(member: BoardMember) {
  editingId.value = member.id
  editName.value = member.name
  editForm.value = formFromProfile(member.profile)
}

async function saveEdit(member: BoardMember) {
  const ok = await run(() => invoke('update_member', {
    id: member.id,
    // A router's name is the board's; only its model list is the human's.
    name: isRouter(member) ? null : (editName.value.trim() || null),
    profile: profileFromForm(editForm.value, member.profile),
  }))
  if (ok) editingId.value = null
}

async function remove(member: BoardMember) {
  await run(() => invoke('delete_member', { id: member.id }))
}

function isRouter(member: BoardMember): boolean {
  return member.id.startsWith(ROUTER_PREFIX)
}

function kindLabel(member: BoardMember): string {
  if (member.kind === 'human') return 'human'
  if (isRouter(member)) return 'router — assigns, takes no work'
  return member.owner_kind === 'human' ? 'your agent' : 'agent'
}

const INPUT = 'w-full rounded-lg border border-white/10 bg-nui-bg px-2 py-1 text-xs text-nui-fg outline-none placeholder:text-nui-muted focus:ring-1 focus:ring-nui-accent'
</script>

<template>
  <div class="nui-scroll flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto" data-testid="board-members">
    <form class="flex max-w-3xl flex-col gap-2 rounded-lg bg-white/[0.03] p-3" @submit.prevent="addAgent">
      <p class="text-xs font-semibold text-nui-fg">Add an agent</p>
      <input v-model="newName" :class="INPUT" placeholder="Name — e.g. Builder" data-testid="member-new-name" :disabled="busy">
      <input v-model="newForm.models" :class="INPUT" placeholder="Models, best first — e.g. qwen3.5:9b, claude-sonnet-5" data-testid="member-new-models" :disabled="busy">
      <input v-model="newForm.capabilities" :class="INPUT" placeholder="Capabilities — e.g. rust, tests, writing" data-testid="member-new-capabilities" :disabled="busy">
      <textarea v-model="newForm.notes" :class="INPUT" rows="2" placeholder="Notes for the router — what this agent is for" :disabled="busy" />
      <div class="flex items-center gap-4">
        <label class="flex items-center gap-2 text-xs text-nui-muted">
          <input v-model="newPersonal" type="checkbox" :disabled="busy">
          Personal — follows you to every board
        </label>
        <span class="min-w-0 flex-1" />
        <button
          type="submit"
          class="rounded-lg bg-nui-accent px-3 py-1 text-xs text-nui-fg disabled:opacity-50"
          :disabled="busy || !newName.trim()"
          data-testid="member-add"
        >
          Add
        </button>
      </div>
      <p v-if="message" class="text-xs text-nui-pink" data-testid="member-message">{{ message }}</p>
    </form>

    <div class="flex max-w-3xl flex-col gap-2">
      <article
        v-for="member in props.roster"
        :key="member.id"
        class="flex flex-col gap-2 rounded-lg bg-white/5 p-3"
        :data-testid="`member-${member.id}`"
      >
        <div class="flex items-center gap-2">
          <span
            class="flex size-6 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold text-nui-bg"
            :class="member.kind === 'human' ? 'bg-nui-accent' : isRouter(member) ? 'bg-nui-yellow' : 'bg-nui-pink'"
          >{{ initials(member.name) }}</span>
          <span class="text-xs text-nui-fg">{{ member.name }}</span>
          <span class="text-[11px] text-nui-muted">{{ member.id }} · {{ kindLabel(member) }}</span>
          <span v-if="member.status === 'busy'" class="text-[11px] text-nui-green">● busy</span>
          <span class="min-w-0 flex-1" />
          <button
            v-if="member.kind === 'agent' && editingId !== member.id"
            type="button"
            class="text-xs text-nui-muted hover:text-nui-fg"
            @click="startEdit(member)"
          >
            Edit
          </button>
          <button
            v-if="member.kind === 'agent' && !isRouter(member)"
            type="button"
            class="text-xs text-nui-muted hover:text-nui-pink"
            :disabled="busy"
            @click="remove(member)"
          >
            Remove
          </button>
        </div>
        <template v-if="editingId === member.id">
          <input v-model="editName" :class="INPUT" :disabled="busy || isRouter(member)" placeholder="Name">
          <input v-model="editForm.models" :class="INPUT" :disabled="busy" placeholder="Models, best first">
          <input v-if="!isRouter(member)" v-model="editForm.capabilities" :class="INPUT" :disabled="busy" placeholder="Capabilities">
          <textarea v-model="editForm.notes" :class="INPUT" rows="2" :disabled="busy" placeholder="Notes for the router" />
          <div class="flex justify-end gap-2">
            <button type="button" class="text-xs text-nui-muted" @click="editingId = null">Cancel</button>
            <button
              type="button"
              class="rounded-lg bg-nui-accent px-3 py-1 text-xs text-nui-fg disabled:opacity-50"
              :disabled="busy"
              @click="saveEdit(member)"
            >
              Save
            </button>
          </div>
        </template>
        <dl v-else-if="member.kind === 'agent'" class="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[11px]">
          <template v-for="(value, label) in { Models: formFromProfile(member.profile).models, Capabilities: formFromProfile(member.profile).capabilities, Notes: formFromProfile(member.profile).notes }" :key="label">
            <template v-if="value">
              <dt class="text-nui-muted">{{ label }}</dt>
              <dd class="break-words text-nui-fg">{{ value }}</dd>
            </template>
          </template>
        </dl>
      </article>
    </div>
  </div>
</template>
