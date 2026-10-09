<template>
  <div class="space-y-6">
    <SettingsSection
      title="Cognitive Memory"
      description="How Nanna recalls and consolidates what it learns over time."
    >
      <template #icon>
        <BrainCircuit class="w-4 h-4 text-nanna-primary" />
      </template>

      <div class="flex items-center justify-end -mt-1">
        <UiBadge variant="secondary">FSRS-6</UiBadge>
      </div>

      <!-- Stats Grid -->
      <div class="grid grid-cols-4 gap-2">
        <div class="p-2 rounded-lg glass-panel text-center">
          <div class="text-lg font-bold text-nanna-success">{{ memoryStats?.active || 0 }}</div>
          <div class="text-xs text-nanna-text-dim">Active</div>
        </div>
        <div class="p-2 rounded-lg glass-panel text-center">
          <div class="text-lg font-bold text-nanna-warning">{{ memoryStats?.dormant || 0 }}</div>
          <div class="text-xs text-nanna-text-dim">Dormant</div>
        </div>
        <div class="p-2 rounded-lg glass-panel text-center">
          <div class="text-lg font-bold text-nanna-text-muted">{{ memoryStats?.silent || 0 }}</div>
          <div class="text-xs text-nanna-text-dim">Silent</div>
        </div>
        <div class="p-2 rounded-lg glass-panel text-center">
          <div class="text-lg font-bold text-nanna-error">{{ memoryStats?.unavailable || 0 }}</div>
          <div class="text-xs text-nanna-text-dim">Faded</div>
        </div>
      </div>

      <!-- Toggles -->
      <div class="flex items-center justify-between">
        <div>
          <div class="text-sm font-medium text-nanna-text">Auto-Remember Messages</div>
          <div class="text-xs text-nanna-text-dim">Save conversations to memory automatically</div>
        </div>
        <UiSwitch :model-value="settings?.auto_remember_messages" label="Auto-remember enabled" @update:model-value="setAutoRememberMessages" />
      </div>

      <!-- Advanced consolidation knobs -->
      <div v-if="showAdvanced" class="space-y-3">
        <div class="p-3 rounded-lg glass-panel">
          <div class="flex items-center justify-between mb-2">
            <span class="text-sm font-medium text-nanna-text">Max Compression</span>
            <span class="text-sm text-nanna-accent font-mono">{{ ((settings?.max_compression_ratio ?? 0.5) * 100).toFixed(0) }}%</span>
          </div>
          <input
            type="range" min="10" max="90" step="5"
            :value="(settings?.max_compression_ratio ?? 0.5) * 100"
            @change="setMaxCompressionRatio(Number(($event.target as HTMLInputElement).value) / 100)"
            class="w-full h-2 glass-well rounded-lg appearance-none cursor-pointer accent-nanna-primary"
          >
          <p class="text-xs text-nanna-text-dim mt-1">Max fraction of memories that can be merged per consolidation run</p>
        </div>
        <div class="p-3 rounded-lg glass-panel">
          <div class="flex items-center justify-between mb-2">
            <span class="text-sm font-medium text-nanna-text">Min Memories Floor</span>
            <span class="text-sm text-nanna-accent font-mono">{{ settings?.min_remaining_memories ?? 20 }}</span>
          </div>
          <input
            type="range" min="5" max="200" step="5"
            :value="settings?.min_remaining_memories ?? 20"
            @change="setMinRemainingMemories(Number(($event.target as HTMLInputElement).value))"
            class="w-full h-2 glass-well rounded-lg appearance-none cursor-pointer accent-nanna-primary"
          >
          <p class="text-xs text-nanna-text-dim mt-1">Never consolidate below this many memories</p>
        </div>
      </div>

      <!-- Dream Button -->
      <UiButton @click="triggerConsolidation" :disabled="consolidating" class="w-full">
        <UiSpinner v-if="consolidating" size="sm" class="mr-2" />
        <Moon v-else class="w-4 h-4 mr-2" />
        {{ consolidating ? 'Dreaming...' : 'Dream Now' }}
      </UiButton>
    </SettingsSection>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { BrainCircuit, Moon } from '@lucide/vue'
import { useSettingsPage } from '~/composables/useSettingsPage'

const store = useSettingsPage()
const { settings, showAdvanced, memoryStats, loadMemoryStats, loadSettings, showToast } = store

// No dreaming switch and no recall-threshold slider: dreaming is a core
// feature with no off switch, and recall keeps its calibrated threshold
// (owner, 2026-09-23). Both controls were no-ops the daemon never saw.
const consolidating = ref(false)

async function setAutoRememberMessages(enabled: boolean) {
  try {
    await invoke('set_auto_remember_messages', { enabled })
    await loadSettings()
  } catch (e: any) {
    showToast(`Could not update auto-remember: ${e.message || e}`, 'error')
  }
}

async function setMaxCompressionRatio(value: number) {
  try {
    await invoke('set_max_compression_ratio', { ratio: value })
    if (settings.value) settings.value.max_compression_ratio = value
  } catch (e: any) {
    showToast(`Could not update compression: ${e.message || e}`, 'error')
  }
}

async function setMinRemainingMemories(value: number) {
  try {
    await invoke('set_min_remaining_memories', { count: value })
    if (settings.value) settings.value.min_remaining_memories = value
  } catch (e: any) {
    showToast(`Could not update memory floor: ${e.message || e}`, 'error')
  }
}

async function triggerConsolidation() {
  consolidating.value = true
  try {
    const result = await invoke<{ memories_processed: number; memories_merged: number }>('trigger_consolidation')
    showToast(`Dreaming complete: ${result.memories_processed} processed`, 'success')
    await loadMemoryStats()
  } catch (e: any) {
    showToast(`Dreaming did not finish: ${e.message || e}`, 'error')
  } finally {
    consolidating.value = false
  }
}
</script>
