<template>
  <div class="flex flex-col h-full relative overflow-hidden">

    <!-- Header -->
    <header class="relative z-10 px-4 sm:px-6 py-3 sm:py-4 border-b border-white/[0.04]">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-base sm:text-lg font-semibold text-white/90">Memory</h2>
          <p class="text-xs sm:text-sm text-white/40">
            {{ semanticMemories.length }} memories
            <span v-if="memoryFilter"> · filtered</span>
          </p>
        </div>

        <div class="flex items-center gap-2">
          <!-- Scope Toggle -->
          <UiGlassTabs v-model="memoryScope" :tabs="scopeItems" size="xs" active-color="accent" />
        </div>
      </div>
    </header>

    <!-- Actions bar -->
    <div class="px-4 sm:px-6 py-3 border-b border-white/[0.04] flex items-center justify-between">
      <div class="flex items-center gap-2">
        <UiSplatterTextarea
          v-model="memoryFilter"
          placeholder="Filter memories..."
          :rows="1"
          class="w-48 sm:w-64"
        />
      </div>
      <div class="flex items-center gap-2">
        <UiGlassButton
          size="xs"
          color="default"
          pill
          :disabled="isDreaming"
          @click="triggerDream"
        >
          {{ isDreaming ? '💭 Dreaming...' : '💭 Dream Now' }}
        </UiGlassButton>
        <UiGlassButton
          v-if="semanticMemories.length > 0"
          size="xs"
          color="danger"
          pill
          @click="clearAllMemories"
        >
          🗑️ Clear All
        </UiGlassButton>
      </div>
    </div>

    <!-- Memory list -->
    <div class="relative z-10 flex-1 min-h-0 flex flex-col overflow-hidden">
      <div class="p-4 sm:p-6 flex-1 min-h-0 flex flex-col">
        <!-- Loading / offline / error / empty -->
        <PageState
          v-if="pageKind"
          :state="pageKind"
          :title="pageTitle"
          :description="pageDescription"
          :primary-action="pagePrimary"
          :primary-busy="isLoading"
          @primary="onPagePrimary"
        />

        <!-- Memories -->
        <VirtualList
          v-else-if="filteredMemories.length > 80"
          :items="filteredMemories"
          :item-height="80"
          :overscan="6"
          class="flex-1 min-h-0"
        >
          <template #default="{ item: memory }">
            <div class="group h-full pb-3">
            <!-- Meta row -->
            <div class="flex items-center justify-between gap-3 mb-1 px-1">
              <div class="flex items-center gap-2 text-[11px] text-white/25 min-w-0">
                <span class="truncate">{{ formatDate(memory.timestamp) }}</span>
                <span v-if="memory.score" class="text-white/15">·</span>
                <span v-if="memory.score" class="shrink-0">{{ (memory.score * 100).toFixed(0) }}% match</span>
                <span v-if="memory.tags?.length" class="text-white/15">·</span>
                <span v-if="memory.tags?.length" class="truncate">{{ memory.tags.join(', ') }}</span>
              </div>
              <div class="flex items-center gap-1 shrink-0">
                <button
                  v-if="editingMemoryId !== memory.id"
                  @click="startEditing(memory)"
                  class="opacity-0 group-hover:opacity-100 text-white/20 hover:text-white/50 transition-all p-1"
                  title="Edit memory"
                >
                  <Pencil class="w-3.5 h-3.5" />
                </button>
                <UiGlassButton
                  v-if="editingMemoryId === memory.id"
                  size="xs"
                  color="accent"
                  pill
                  @click="saveEditing(memory.id)"
                >
                  Save
                </UiGlassButton>
                <UiGlassButton
                  v-if="editingMemoryId === memory.id"
                  size="xs"
                  color="default"
                  pill
                  @click="cancelEditing"
                >
                  Cancel
                </UiGlassButton>
                <button
                  @click="deleteMemory(memory.id)"
                  class="opacity-0 group-hover:opacity-100 text-white/20 hover:text-red-400/60 transition-all p-1"
                  title="Delete memory"
                >
                  <Trash2 class="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
            <!-- Rich text editor -->
            <MemoryEditor
              :model-value="editingMemoryId === memory.id ? editBuffers[memory.id] : memory.content"
              :editable="editingMemoryId === memory.id"
              placeholder="Write a memory..."
              @update:model-value="editBuffers[memory.id] = $event"
            />
            </div>
          </template>
        </VirtualList>
        <div v-else class="space-y-3 flex-1 min-h-0 overflow-y-auto">
          <div
            v-for="memory in filteredMemories"
            :key="memory.id"
            class="group"
          >
            <!-- Meta row -->
            <div class="flex items-center justify-between gap-3 mb-1 px-1">
              <div class="flex items-center gap-2 text-[11px] text-white/25 min-w-0">
                <span class="truncate">{{ formatDate(memory.timestamp) }}</span>
                <span v-if="memory.score" class="text-white/15">·</span>
                <span v-if="memory.score" class="shrink-0">{{ (memory.score * 100).toFixed(0) }}% match</span>
                <span v-if="memory.tags?.length" class="text-white/15">·</span>
                <span v-if="memory.tags?.length" class="truncate">{{ memory.tags.join(', ') }}</span>
              </div>
              <div class="flex items-center gap-1 shrink-0">
                <button
                  v-if="editingMemoryId !== memory.id"
                  @click="startEditing(memory)"
                  class="opacity-0 group-hover:opacity-100 text-white/20 hover:text-white/50 transition-all p-1"
                  title="Edit memory"
                >
                  <Pencil class="w-3.5 h-3.5" />
                </button>
                <UiGlassButton
                  v-if="editingMemoryId === memory.id"
                  size="xs"
                  color="accent"
                  pill
                  @click="saveEditing(memory.id)"
                >
                  Save
                </UiGlassButton>
                <UiGlassButton
                  v-if="editingMemoryId === memory.id"
                  size="xs"
                  color="default"
                  pill
                  @click="cancelEditing"
                >
                  Cancel
                </UiGlassButton>
                <button
                  @click="deleteMemory(memory.id)"
                  class="opacity-0 group-hover:opacity-100 text-white/20 hover:text-red-400/60 transition-all p-1"
                  title="Delete memory"
                >
                  <Trash2 class="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
            <!-- Rich text editor -->
            <MemoryEditor
              :model-value="editingMemoryId === memory.id ? editBuffers[memory.id] : memory.content"
              :editable="editingMemoryId === memory.id"
              placeholder="Write a memory..."
              @update:model-value="editBuffers[memory.id] = $event"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch, inject } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { Globe, FolderKanban, Trash2, Pencil } from '@lucide/vue'

const { status: backendStatus, isOnline } = useBackend()
const { confirm } = useConfirm()
const toast = useToast()

// Scope tabs
const currentWorkspace = inject<any>('currentWorkspace', ref(null))
const memoryScope = ref<string>('global')

const scopeItems = computed(() => {
  const items: { id: string; label: string; icon?: any }[] = [
    { id: 'global', label: 'Global', icon: Globe },
  ]
  if (currentWorkspace?.value) {
    items.push({ id: 'workspace', label: currentWorkspace.value.name || 'Workspace', icon: FolderKanban })
  } else {
    items.push({ id: 'workspace', label: 'Workspace', icon: FolderKanban })
  }
  return items
})

// State
const memoryFilter = ref('')
const isLoading = ref(true)
const loadError = ref<string | null>(null)
const isDreaming = ref(false)
const editingMemoryId = ref<string | null>(null)
const editBuffers = reactive<Record<string, string>>({})
const semanticMemories = ref<any[]>([])

const pageKind = computed(() => {
  if (isLoading.value) return 'loading' as const
  if (!isOnline.value) return 'offline' as const
  if (loadError.value) return 'error' as const
  if (filteredMemories.value.length === 0) return 'empty' as const
  return null
})
const pageTitle = computed(() => {
  if (pageKind.value === 'loading') return 'Loading memories…'
  if (pageKind.value === 'offline') return 'Daemon offline'
  if (pageKind.value === 'error') return 'Could not load memories'
  return memoryFilter.value ? 'No matching memories' : 'No memories yet'
})
const pageDescription = computed(() => {
  if (pageKind.value === 'loading') return 'Pulling the cognitive store from the daemon.'
  if (pageKind.value === 'offline') return 'Memory is served by the daemon control plane. Reconnect to browse or dream.'
  if (pageKind.value === 'error') return loadError.value || 'Unknown error'
  return memoryFilter.value
    ? 'Try a different filter.'
    : 'Memories are created automatically from conversations and can be added manually.'
})
const pagePrimary = computed(() => {
  if (pageKind.value === 'offline' || pageKind.value === 'error') return 'Retry'
  return ''
})
function onPagePrimary() {
  void fetchMemories()
}


function startEditing(memory: any) {
  editingMemoryId.value = memory.id
  editBuffers[memory.id] = memory.content
}

function cancelEditing() {
  editingMemoryId.value = null
}

async function saveEditing(id: string) {
  const content = editBuffers[id]
  if (content == null) return
  try {
    await invoke('update_memory', { id, content })
    const mem = semanticMemories.value.find((m: any) => m.id === id)
    if (mem) mem.content = content
    toast.success('Memory saved')
  } catch (e) {
    console.error('Failed to update memory:', e)
    toast.error('Save failed', e instanceof Error ? e.message : String(e))
  }
  editingMemoryId.value = null
}

const filteredMemories = computed(() => {
  if (!memoryFilter.value) return semanticMemories.value
  const q = memoryFilter.value.toLowerCase()
  return semanticMemories.value.filter((m: any) =>
    m.content?.toLowerCase().includes(q)
  )
})

/** The latest fetch; a slower earlier one (another scope) must not land over it. */
let fetchTicket = 0

async function fetchMemories() {
  const ticket = ++fetchTicket
  isLoading.value = true
  loadError.value = null
  try {
    const res = await invoke('list_memories', {
      scope: memoryScope.value,
      workspaceId: currentWorkspace?.value?.id,
    })
    if (ticket !== fetchTicket) return
    semanticMemories.value = res as any[]
  } catch (e) {
    if (ticket !== fetchTicket) return
    console.error('Failed to fetch memories:', e)
    loadError.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (ticket === fetchTicket) isLoading.value = false
  }
}

async function deleteMemory(id: string) {
  const ok = await confirm({
    title: 'Delete memory?',
    message: 'This removes the memory from the cognitive store. It cannot be undone.',
    confirmLabel: 'Delete',
    danger: true,
  })
  if (!ok) return
  try {
    await invoke('delete_memory', { id })
    semanticMemories.value = semanticMemories.value.filter((m: any) => m.id !== id)
    toast.success('Memory deleted')
  } catch (e) {
    console.error('Failed to delete memory:', e)
    toast.error('Delete failed', e instanceof Error ? e.message : String(e))
  }
}

async function triggerDream() {
  isDreaming.value = true
  try {
    await invoke('trigger_consolidation')
    await fetchMemories()
    toast.success('Dream cycle finished')
  } catch (e) {
    console.error('Dream failed:', e)
    toast.error('Dream failed', e instanceof Error ? e.message : String(e))
  } finally {
    isDreaming.value = false
  }
}

async function clearAllMemories() {
  const ok = await confirm({
    title: 'Clear all memories?',
    message: 'This permanently clears memories in the current scope. It cannot be undone.',
    confirmLabel: 'Clear all',
    danger: true,
  })
  if (!ok) return
  try {
    await invoke('clear_memories', {
      scope: memoryScope.value,
      workspaceId: currentWorkspace?.value?.id,
    })
    semanticMemories.value = []
    toast.success('Memories cleared')
  } catch (e) {
    console.error('Failed to clear memories:', e)
    toast.error('Clear failed', e instanceof Error ? e.message : String(e))
  }
}

function formatDate(ts: string | number): string {
  if (!ts) return ''
  const d = new Date(typeof ts === 'number' ? ts * 1000 : ts)
  const now = new Date()
  const diff = now.getTime() - d.getTime()
  if (diff < 60000) return 'just now'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
  if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`
  return d.toLocaleDateString()
}

watch(memoryScope, () => {
  fetchMemories()
})

onMounted(() => {
  fetchMemories()
})
</script>
