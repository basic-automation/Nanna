<template>
  <div class="flex flex-col h-full">
    <!-- Header -->
    <header class="px-4 sm:px-6 py-3 sm:py-4 border-b border-white/[0.04]">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-base sm:text-lg font-semibold text-nanna-text">Scheduler</h2>
          <p class="text-xs sm:text-sm text-nanna-text-muted">
            Manage cron jobs and scheduled tasks
          </p>
        </div>
        <div class="flex items-center gap-2">
          <UiSwitch :model-value="schedulerEnabled" label="Scheduler enabled" @update:modelValue="toggleScheduler" />
          <span class="text-sm text-nanna-text-muted">{{ schedulerEnabled ? 'Enabled' : 'Disabled' }}</span>
          <UiButton @click="openCreateModal" size="sm" :disabled="!schedulerEnabled">
            ➕ New Job
          </UiButton>
        </div>
      </div>
    </header>

    <!-- Content -->
    <div class="flex-1 overflow-y-auto p-4 sm:p-6">
      <!-- Loading state -->
            <PageState
        v-if="loading || !isOnline || loadError || jobs.length === 0"
        :state="loading ? 'loading' : (!isOnline ? 'offline' : (loadError ? 'error' : 'empty'))"
        :title="loading ? 'Loading scheduler…' : (!isOnline ? 'Daemon offline' : (loadError ? 'Could not load jobs' : 'No scheduled jobs'))"
        :description="loading
          ? 'Reading cron jobs from the daemon scheduler.'
          : (!isOnline
            ? 'The scheduler lives in the daemon. Reconnect to create or inspect jobs.'
            : (loadError || 'Create cron jobs to run tasks on a schedule. Jobs can trigger prompts, run tools, or send channel messages.'))"
        :primary-action="loading ? '' : ((!isOnline || loadError) ? 'Retry' : 'Create job')"
        :primary-busy="loading"
        @primary="onSchedulerPrimary"
      />

      <!-- Job list -->
      <div v-else class="space-y-3 sm:space-y-4">
        <UiCard
          v-for="job in jobs"
          :key="job.id"
          class="p-4 hover:bg-white/[0.03] transition-colors"
          :class="{ 'opacity-50': !job.enabled }"
        >
          <div class="flex items-start justify-between gap-4">
            <div class="min-w-0 flex-1">
              <div class="flex items-center gap-2 mb-1">
                <span class="font-semibold text-nanna-text">{{ job.name }}</span>
                <UiBadge v-if="!job.enabled" variant="warning" size="sm">disabled</UiBadge>
                <UiBadge v-if="job.name === 'heartbeat'" variant="secondary" size="sm">system</UiBadge>
                <UiBadge v-if="job.name === 'memory_consolidation'" variant="secondary" size="sm">dreaming</UiBadge>
              </div>
              
              <div class="flex items-center gap-4 text-sm text-nanna-text-muted mb-2">
                <span class="font-mono text-nanna-accent">{{ job.schedule }}</span>
                <span>{{ job.schedule_description }}</span>
              </div>
              
              <p v-if="job.payload" class="text-sm text-nanna-text-dim mb-2 line-clamp-2">
                {{ job.payload }}
              </p>
              
              <div class="flex items-center gap-4 text-xs text-nanna-text-dim">
                <span v-if="job.last_run">Last: {{ formatDate(job.last_run) }}</span>
                <span v-if="job.next_run">Next: {{ formatDate(job.next_run) }}</span>
                <span>Runs: {{ job.run_count }}</span>
              </div>
            </div>
            
            <div class="flex items-center gap-2">
              <UiButton 
                @click="runNow(job)" 
                variant="ghost" 
                size="sm"
                :disabled="!schedulerEnabled"
                title="Run now"
              >
                ▶️
              </UiButton>
              <UiButton 
                @click="viewHistory(job)" 
                variant="ghost" 
                size="sm"
                title="View history"
              >
                📋
              </UiButton>
              <UiButton 
                @click="toggleJobEnabled(job)" 
                variant="ghost" 
                size="sm"
                :title="job.enabled ? 'Disable' : 'Enable'"
              >
                {{ job.enabled ? '⏸️' : '▶️' }}
              </UiButton>
              <UiButton 
                v-if="!isSystemJob(job)"
                @click="editJob(job)" 
                variant="ghost" 
                size="sm"
              >
                ✏️
              </UiButton>
              <UiButton 
                v-if="!isSystemJob(job)"
                @click="confirmDelete(job)" 
                variant="ghost" 
                size="sm" 
                class="text-red-400 hover:text-red-300"
              >
                🗑️
              </UiButton>
            </div>
          </div>
        </UiCard>
      </div>
    </div>

    <!-- Create/Edit Modal -->
    <UiModal v-model="showModal" :title="editing ? 'Edit Job' : 'Create Job'" size="lg">
      <div class="space-y-4">
        <!-- Name -->
        <div>
          <label class="block text-sm font-medium text-nanna-text mb-1">Name</label>
          <UiInput
            v-model="form.name"
            placeholder="daily-backup"
            :disabled="editing"
          />
        </div>

        <!-- Schedule -->
        <div>
          <label class="block text-sm font-medium text-nanna-text mb-1">Schedule (cron expression)</label>
          <UiInput
            v-model="form.schedule"
            placeholder="0 8 * * *"
            class="font-mono"
            @input="validateSchedule"
          />
          <p v-if="scheduleValidation.valid" class="text-xs text-nanna-success mt-1">
            ✓ {{ scheduleValidation.description }}
          </p>
          <p v-else-if="scheduleValidation.description" class="text-xs text-nanna-error mt-1">
            ✗ {{ scheduleValidation.description }}
          </p>
          <p class="text-xs text-nanna-text-dim mt-1">
            Format: minute hour day month weekday (e.g., "0 8 * * *" = 8 AM daily)
          </p>
        </div>

        <!-- Quick presets -->
        <div>
          <label class="block text-sm font-medium text-nanna-text mb-1">Quick Presets</label>
          <div class="flex flex-wrap gap-2">
            <UiButton 
              v-for="preset in schedulePresets" 
              :key="preset.value"
              variant="secondary"
              size="sm"
              @click="applyPreset(preset)"
            >
              {{ preset.label }}
            </UiButton>
          </div>
        </div>

        <!-- Payload -->
        <div>
          <label class="block text-sm font-medium text-nanna-text mb-1">Payload / Prompt</label>
          <textarea
            v-model="form.payload"
            rows="4"
            class="w-full glass border border-nanna-border rounded-md px-3 py-2 text-sm text-nanna-text placeholder-nanna-text-dim focus:ring-2 focus:ring-nanna-primary/50 focus:border-nanna-primary"
            placeholder="What should Nanna do when this job runs?"
          />
        </div>

        <!-- Timezone -->
        <div>
          <label class="block text-sm font-medium text-nanna-text mb-1">Timezone</label>
          <UiSelect v-model="form.timezone" :options="timezoneOptions" />
        </div>
      </div>

      <template #footer>
        <div class="flex justify-end gap-2">
          <UiButton variant="ghost" @click="showModal = false">Cancel</UiButton>
          <UiButton 
            @click="saveJob" 
            :disabled="!scheduleValidation.valid || !form.name"
          >
            {{ editing ? 'Save Changes' : 'Create Job' }}
          </UiButton>
        </div>
      </template>
    </UiModal>

    <!-- History Modal -->
    <UiModal v-model="showHistoryModal" title="Job Run History" size="lg">
      <div v-if="historyLoading" class="text-center py-8 text-nanna-text-muted">
        Loading history...
      </div>
      <div v-else-if="history.length === 0" class="text-center py-8 text-nanna-text-muted">
        No runs recorded yet.
      </div>
      <div v-else class="space-y-2 max-h-96 overflow-y-auto">
        <div 
          v-for="run in history" 
          :key="run.id"
          class="p-3 glass-well rounded-md"
        >
          <div class="flex items-center justify-between mb-1">
            <span :class="run.success ? 'text-nanna-success' : 'text-nanna-error'">
              {{ run.success ? '✓ Success' : '✗ Failed' }}
            </span>
            <span class="text-xs text-nanna-text-dim">
              {{ formatDate(run.started_at) }}
            </span>
          </div>
          <div v-if="run.output" class="text-sm text-nanna-text-muted mt-1">
            {{ run.output }}
          </div>
          <div v-if="run.error" class="text-sm text-nanna-error mt-1">
            {{ run.error }}
          </div>
        </div>
      </div>
    </UiModal>

    <!-- Delete Confirmation -->
    <UiModal v-model="showDeleteModal" title="Delete Job?" size="sm">
      <p class="text-nanna-text-muted">
        Are you sure you want to delete "{{ jobToDelete?.name }}"?
        This action cannot be undone.
      </p>
      <template #footer>
        <div class="flex justify-end gap-2">
          <UiButton variant="ghost" @click="showDeleteModal = false">Cancel</UiButton>
          <UiButton variant="destructive" @click="deleteJob">Delete</UiButton>
        </div>
      </template>
    </UiModal>
  </div>
</template>

<script setup lang="ts">
import { invoke } from '@tauri-apps/api/core'

const { isOnline } = useBackend()
const toast = useToast()
const { confirm } = useConfirm()

interface CronJob {
  id: string
  name: string
  schedule: string
  schedule_description: string
  payload: string
  enabled: boolean
  last_run: string | null
  next_run: string | null
  run_count: number
  timezone: string
}

interface JobRun {
  id: number
  job_id: string
  started_at: string
  finished_at: string | null
  success: boolean
  output: string | null
  error: string | null
  duration_ms: number | null
}

const loading = ref(true)
const loadError = ref<string | null>(null)
const jobs = ref<CronJob[]>([])
const schedulerEnabled = ref(true)

const showModal = ref(false)
const editing = ref(false)
/** The job being edited, by id: two jobs may share a name. */
const editingId = ref<string | null>(null)
const form = ref({
  name: '',
  schedule: '',
  payload: '',
  timezone: 'UTC',
})

const scheduleValidation = ref({
  valid: false,
  description: '',
})

const showHistoryModal = ref(false)
const historyLoading = ref(false)
const history = ref<JobRun[]>([])
const selectedJob = ref<CronJob | null>(null)

const showDeleteModal = ref(false)
const jobToDelete = ref<CronJob | null>(null)

// UiSelect renders from this prop. It used to be given slotted <option>
// children instead, which the component never reads — so the timezone
// dropdown rendered empty.
const timezoneOptions = [
  { value: 'UTC', label: 'UTC' },
  { value: 'America/New_York', label: 'America/New_York' },
  { value: 'America/Chicago', label: 'America/Chicago' },
  { value: 'America/Denver', label: 'America/Denver' },
  { value: 'America/Los_Angeles', label: 'America/Los_Angeles' },
  { value: 'Europe/London', label: 'Europe/London' },
  { value: 'Europe/Paris', label: 'Europe/Paris' },
  { value: 'Asia/Tokyo', label: 'Asia/Tokyo' },
]

const schedulePresets = [
  { label: 'Every 5 min', value: '*/5 * * * *' },
  { label: 'Hourly', value: '0 * * * *' },
  { label: '8 AM daily', value: '0 8 * * *' },
  { label: '9 PM daily', value: '0 21 * * *' },
  { label: 'Weekdays 9 AM', value: '0 9 * * 1-5' },
  { label: 'Weekly (Sun)', value: '0 0 * * 0' },
  { label: 'Monthly (1st)', value: '0 0 1 * *' },
]

onMounted(async () => {
  await loadJobs()
  await loadSchedulerState()
})

function onSchedulerPrimary() {
  if (!isOnline.value || loadError.value) {
    void loadJobs()
    return
  }
  openCreateModal()
}


async function loadJobs() {
  loading.value = true
  loadError.value = null
  try {
    jobs.value = await invoke<CronJob[]>('list_cron_jobs')
  } catch (e) {
    console.error('Failed to load jobs:', e)
  } finally {
    loading.value = false
  }
}

async function loadSchedulerState() {
  try {
    const settings = await invoke<any>('get_extended_settings')
    schedulerEnabled.value = settings.scheduler_enabled
  } catch (e) {
    console.error('Failed to load scheduler state:', e)
  }
}

// One-way bound: the switch shows what the daemon accepted. With v-model it
// flipped on the click, so a refused save left it showing the wrong state.
async function toggleScheduler(enabled: boolean) {
  try {
    await invoke('set_scheduler_enabled', { enabled })
    schedulerEnabled.value = enabled
  } catch (e) {
    console.error('Failed to toggle scheduler:', e)
    toast.show(`Could not ${enabled ? 'enable' : 'disable'} the scheduler`, 'error', String(e))
  }
}

function openCreateModal() {
  editing.value = false
  form.value = { name: '', schedule: '', payload: '', timezone: 'UTC' }
  scheduleValidation.value = { valid: false, description: '' }
  showModal.value = true
}

function editJob(job: CronJob) {
  editing.value = true
  editingId.value = job.id
  form.value = {
    name: job.name,
    schedule: job.schedule,
    payload: job.payload,
    timezone: job.timezone,
  }
  validateSchedule()
  showModal.value = true
}

function applyPreset(preset: { value: string }) {
  form.value.schedule = preset.value
  validateSchedule()
}

async function validateSchedule() {
  if (!form.value.schedule) {
    scheduleValidation.value = { valid: false, description: '' }
    return
  }
  
  try {
    const [valid, description] = await invoke<[boolean, string]>('validate_cron_expression', {
      expression: form.value.schedule,
    })
    scheduleValidation.value = { valid, description }
  } catch (e) {
    scheduleValidation.value = { valid: false, description: 'Validation error' }
  }
}

async function saveJob() {
  try {
    if (editing.value) {
      // The prompt is part of the edit: only the schedule used to be sent,
      // so a changed prompt was dropped while the modal closed as if saved.
      if (editingId.value) {
        await invoke('update_cron_job', {
          jobId: editingId.value,
          schedule: form.value.schedule,
          payload: form.value.payload,
        })
      }
    } else {
      await invoke('create_cron_job', {
        name: form.value.name,
        schedule: form.value.schedule,
        payload: form.value.payload,
        timezone: form.value.timezone,
      })
    }
    showModal.value = false
    await loadJobs()
  } catch (e) {
    console.error('Failed to save job:', e)
  }
}

async function toggleJobEnabled(job: CronJob) {
  try {
    await invoke('set_cron_job_enabled', {
      jobId: job.id,
      enabled: !job.enabled,
    })
    await loadJobs()
  } catch (e) {
    console.error('Failed to toggle job:', e)
  }
}

async function runNow(job: CronJob) {
  try {
    await invoke('run_cron_job_now', { jobId: job.id })
    await loadJobs()
  } catch (e) {
    console.error('Failed to run job:', e)
  }
}

async function viewHistory(job: CronJob) {
  selectedJob.value = job
  historyLoading.value = true
  showHistoryModal.value = true
  
  try {
    history.value = await invoke<JobRun[]>('get_cron_job_history', {
      jobId: job.id,
      limit: 20,
    })
  } catch (e) {
    console.error('Failed to load history:', e)
    history.value = []
  } finally {
    historyLoading.value = false
  }
}

function confirmDelete(job: CronJob) {
  jobToDelete.value = job
  showDeleteModal.value = true
}

async function deleteJob() {
  if (!jobToDelete.value) return
  
  try {
    await invoke('delete_cron_job', { jobId: jobToDelete.value.id })
    showDeleteModal.value = false
    jobToDelete.value = null
    await loadJobs()
  } catch (e) {
    console.error('Failed to delete job:', e)
  }
}

function isSystemJob(job: CronJob): boolean {
  return job.name === 'heartbeat' || job.name === 'memory_consolidation'
}

function formatDate(date: string | null): string {
  if (!date) return 'N/A'
  return new Date(date).toLocaleString()
}


</script>
